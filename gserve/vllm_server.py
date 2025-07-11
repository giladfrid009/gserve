import os
import signal
import sys
import argparse
import json
import gc
import asyncio
import logging
import uvicorn
from typing import List, Dict, Optional

from gserve.log_utils import LOG_LEVEL_ENV, setup_logging

from gserve.schema import ChatRequest, GenerateRequest, ResponseOutput
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Response
import msgspec

import ray
import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

setup_logging()  # Configure logging based on the environment
logger = logging.getLogger(__name__)

# globals
_llm_instance: Optional[LLM] = None
_lora_request: Optional[LoRARequest] = None


@asynccontextmanager
async def release_resources(app: FastAPI):
    """Cleans up resources when the FastAPI app shuts down."""

    global _llm_instance

    try:
        yield  # App is running
    finally:

        if _llm_instance is None:
            # server was never started or already cleaned up
            return

        try:
            destroy_model_parallel()
            destroy_distributed_environment()

            del _llm_instance
            _llm_instance = None

            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()

            logger.info("Resources released successfully.")

        except Exception as e:
            logger.exception("Error during cleanup: %s", e)


app = FastAPI(title="vLLM Batched-Chat & Generate Server", lifespan=release_resources)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health-check. Returns 200 OK if the server is up.
    """
    return {"status": "ok"}


@app.post("/chat")
async def chat_endpoint(request: Request) -> Response:
    """
    Batched chat endpoint. Expects JSON matching ChatRequest, calls `llm.chat(...)`
    on the entire batch, and returns a list of M generated strings.
    """
    if _llm_instance is None:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    data = await request.body()
    try:
        req = msgspec.json.decode(data, type=ChatRequest)
        logger.debug("/chat with %d conversation(s)", len(req.conversations))
    except msgspec.DecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    try:
        req_outputs = _llm_instance.chat(
            req.conversations,
            sampling_params=req.params,
            lora_request=_lora_request,
            add_generation_prompt=True,
            continue_final_message=False,
            use_tqdm=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during vLLM inference: {e!r}",
        )

    responses: List[List[ResponseOutput]] = []
    for res in req_outputs:
        gens: List[ResponseOutput] = []
        for o in res.outputs:
            gens.append(ResponseOutput(text=o.text, logprobs=o.logprobs))
        responses.append(gens)
    return Response(
        content=msgspec.json.encode(responses),
        media_type="application/json",
    )


@app.post("/generate")
async def generate_endpoint(request: Request) -> Response:
    """
    Batched generate endpoint. Expects JSON matching GenerateRequest, calls
    `llm.generate(...)` on the entire batch of N prompts, and returns a list of N strings.
    """
    if _llm_instance is None:
        raise HTTPException(status_code=500, detail="LLM not initialized.")

    data = await request.body()
    try:
        req = msgspec.json.decode(data, type=GenerateRequest)
        logger.debug("/generate with %d prompt(s)", len(req.prompts))
    except msgspec.DecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    try:
        req_outputs = _llm_instance.generate(
            req.prompts,
            sampling_params=req.params,
            lora_request=_lora_request,
            use_tqdm=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during vLLM inference: {e!r}",
        )

    responses: List[List[ResponseOutput]] = []
    for res in req_outputs:
        gens: List[ResponseOutput] = []
        for o in res.outputs:
            gens.append(ResponseOutput(text=o.text, logprobs=o.logprobs))
        responses.append(gens)
    return Response(
        content=msgspec.json.encode(responses),
        media_type="application/json",
    )


@app.post("/shutdown")
async def shutdown_endpoint() -> Response:
    """Shut down the server after cleaning up resources.

    The response is sent before the process receives ``SIGTERM`` so the
    client gets confirmation that the shutdown request was processed.
    """
    loop = asyncio.get_running_loop()
    loop.call_later(0.1, os.kill, os.getpid(), signal.SIGTERM)
    return Response(content="shutting down", media_type="text/plain")


def server_main(
    model_name: str,
    host: str,
    port: int,
    gpus: str,
    llm_kwargs: Optional[str] = None,
    lora_path: Optional[str] = None,
) -> None:
    """
    Entrypoint to run the FastAPI server. Called when this file is run with
    ``--serve``.

    1) Sets ``CUDA_VISIBLE_DEVICES`` so only the given GPUs are visible.
    2) Initializes a single :class:`vllm.LLM` with ``model_name``.
    3) If ``lora_path`` is provided, loads the LoRA adapter and enables LoRA.
    4) Launches ``uvicorn.Server`` at ``host:port``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    global _llm_instance, _lora_request
    extra = json.loads(llm_kwargs) if llm_kwargs else {}
    if lora_path is not None:
        _lora_request = LoRARequest("lora_adapter", 1, lora_path=lora_path)
        extra.setdefault("enable_lora", True)

    logger.info("Initializing LLM '%s' on GPUs %s", model_name, gpus)

    _llm_instance = LLM(
        model=model_name,
        **extra,
    )

    logger.info("LLM initialized, starting server at %s:%s", host, port)

    log_level = logging.getLevelName(logging.getLogger().getEffectiveLevel()).lower()
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Batched Chat & Generate Server")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as FastAPI server (invokes server_main).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID or local path (e.g. 'meta-llama/Llama-3.2-1b-Instruct').",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP to bind the FastAPI server to.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the FastAPI server to.")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU indices visible to this process (e.g. '0' or '0,1').",
    )
    parser.add_argument(
        "--llm_kwargs",
        type=str,
        default=None,
        help="JSON string with additional arguments passed to vllm.LLM",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to a LoRA adapter to use for inference",
    )
    args = parser.parse_args()

    if args.serve:
        if args.model is None:
            print("ERROR: --model must be specified when using --serve", file=sys.stderr)
            sys.exit(1)

        server_main(
            model_name=args.model,
            host=args.host,
            port=args.port,
            gpus=args.gpus,
            llm_kwargs=args.llm_kwargs,
            lora_path=args.lora_path,
        )

        sys.exit(0)

    # If not invoked with --serve, do nothing (this file is meant to be imported or launched with --serve).
