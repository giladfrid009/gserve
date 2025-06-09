import requests
from requests.exceptions import RequestException
from vllm import SamplingParams
from typing import List, Dict, Optional
import msgspec

from src.vllm_server import (
    ResponseOutput,
    ChatRequest,
    GenerateRequest,
)


class VLLMClient:
    """HTTP client for the :mod:`vllm_server` FastAPI service."""

    def __init__(self, host: str, port: int, timeout: Optional[float] = 30.0):
        """
        Args:
          host: Host where the vLLM server is running (e.g. "127.0.0.1").
          port: Port where the vLLM server listens (e.g. 8000).
          timeout: HTTP request timeout in seconds. ``None`` disables timeouts.
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

        # Basic health check
        health_url = f"{self.base_url}/health"
        try:
            resp = requests.get(health_url, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Health check returned HTTP {resp.status_code}")
        except Exception as e:
            raise RuntimeError(f"Cannot reach vLLM server at {health_url}: {e!r}")

    def chat(
        self,
        conversations: List[List[Dict[str, str]]],
        sampling_params: SamplingParams,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """
        Send a batch of M conversations to POST /chat, return a list of M generated strings.

        Raises RuntimeError on HTTP errors, non-200 responses, or malformed JSON.
        """
        url = f"{self.base_url}/chat"

        payload = ChatRequest(
            conversations=conversations,
            params=sampling_params,
        )

        try:
            # ``requests.post(json=...)`` would re-encode using ``json.dumps``.
            # Here we pre-encode with msgspec for speed and pass the raw bytes.
            response = requests.post(
                url,
                data=msgspec.json.encode(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except RequestException as e:
            raise RuntimeError(f"Failed to POST /chat → {e!r}")

        if response.status_code != 200:
            raise RuntimeError(f"/chat returned HTTP {response.status_code}: {response.text}")

        try:
            output = msgspec.json.decode(response.content, type=list[list[ResponseOutput]])
        except msgspec.DecodeError as e:
            raise RuntimeError(f"Invalid JSON in /chat response: {e!r}")

        if return_extra:
            return output
        return [[o.text for o in outs] for outs in output]

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """
        Send a batch of N prompts to POST /generate, return a list of N generated strings.

        Raises RuntimeError on HTTP errors, non-200 responses, or malformed JSON.
        """
        url = f"{self.base_url}/generate"

        payload = GenerateRequest(
            prompts=prompts,
            params=sampling_params,
        )

        try:
            # Pre-encode with msgspec rather than letting ``requests`` call
            # ``json.dumps`` internally.
            response = requests.post(
                url,
                data=msgspec.json.encode(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except RequestException as e:
            raise RuntimeError(f"Failed to POST /generate → {e!r}")

        if response.status_code != 200:
            raise RuntimeError(f"/generate returned HTTP {response.status_code}: {response.text}")

        try:
            output = msgspec.json.decode(response.content, type=list[list[ResponseOutput]])
        except msgspec.DecodeError as e:
            raise RuntimeError(f"Invalid JSON in /generate response: {e!r}")

        if return_extra:
            return output
        return [[o.text for o in outs] for outs in output]
