import asyncio
from typing import List, Dict, Optional

import httpx
from httpx import HTTPError
from vllm import SamplingParams
import msgspec

from src.vllm_server import ResponseOutput, ChatRequest, GenerateRequest


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
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

        # Basic health check using a short-lived sync request so failures don't
        # leak open connections on ``_client``.
        health_url = f"{self.base_url}/health"
        try:
            resp = httpx.get(health_url, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Health check returned HTTP {resp.status_code}")
        except Exception as e:
            asyncio.run(self._client.aclose())
            raise RuntimeError(f"Cannot reach vLLM server at {health_url}: {e!r}")

    async def chat_async(
        self,
        conversations: List[List[Dict[str, str]]],
        sampling_params: SamplingParams | None,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """Call the ``/chat`` endpoint asynchronously."""
        url = f"{self.base_url}/chat"

        payload = ChatRequest(
            conversations=conversations,
            params=sampling_params,
        )

        try:
            response = await self._client.post(
                url,
                content=msgspec.json.encode(payload),
                headers={"Content-Type": "application/json"},
            )
        except HTTPError as e:
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

    async def generate_async(
        self,
        prompts: List[str],
        sampling_params: SamplingParams | None,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """Call the ``/generate`` endpoint asynchronously."""
        url = f"{self.base_url}/generate"

        payload = GenerateRequest(
            prompts=prompts,
            params=sampling_params,
        )

        try:
            response = await self._client.post(
                url,
                content=msgspec.json.encode(payload),
                headers={"Content-Type": "application/json"},
            )
        except HTTPError as e:
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

    def chat(
        self,
        conversations: List[List[Dict[str, str]]],
        sampling_params: SamplingParams | None,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """Synchronous wrapper over :meth:`chat_async`."""
        return asyncio.run(self.chat_async(conversations, sampling_params, return_extra))

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams | None,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """Synchronous wrapper over :meth:`generate_async`."""
        return asyncio.run(self.generate_async(prompts, sampling_params, return_extra))

    def close(self) -> None:
        """Synchronous wrapper over :meth:`aclose`."""
        asyncio.run(self.aclose())

    async def aclose(self) -> None:
        """Close the underlying asynchronous HTTP client."""
        await self._client.aclose()
