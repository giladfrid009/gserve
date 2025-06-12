import requests
from requests.exceptions import RequestException
from vllm import SamplingParams
from typing import List, Dict, Optional, Literal, overload
import msgspec
import logging

from gserve.schema import ChatRequest, GenerateRequest, ResponseOutput

logger = logging.getLogger(__name__)


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
        self.session = requests.Session()

        # Basic health check
        health_url = f"{self.base_url}/health"
        try:
            resp = self.session.get(health_url, timeout=self.timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Health check returned HTTP {resp.status_code}")
        except Exception as e:
            raise RuntimeError(f"Cannot reach vLLM server at {health_url}: {e!r}")

    @overload
    def chat(
        self,
        conversations: List[List[Dict[str, str]]],
        sampling_params: SamplingParams | None,
        return_extra: Literal[False],
    ) -> List[List[str]]: ...

    @overload
    def chat(
        self,
        conversations: List[List[Dict[str, str]]],
        sampling_params: SamplingParams | None,
        return_extra: Literal[True],
    ) -> List[List[ResponseOutput]]: ...

    def chat(
        self,
        conversations: List[List[Dict[str, str]]],
        sampling_params: SamplingParams | None,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """
        Send a batched chat request to the vLLM server.

        Args:
            conversations: List of conversations, where each conversation is a list
                of message dictionaries with "role" and "content" keys.
            sampling_params: Optional sampling parameters for the generation.
            return_extra: If True, return the full `ResponseOutput` objects instead
                of just the generated text.

        Returns:
            List of lists containing generated responses. If `return_extra` is True,
            returns a list of lists of ResponseOutput objects, otherwise just the text.
        """
        payload = ChatRequest(
            conversations=conversations,
            params=sampling_params,
        )

        try:
            # ``requests.post(json=...)`` would re-encode using ``json.dumps``.
            # Here we pre-encode with msgspec for speed and pass the raw bytes.
            response = self.session.post(
                url=f"{self.base_url}/chat",
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

    @overload
    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams | None,
        return_extra: Literal[False],
    ) -> List[List[str]]: ...

    @overload
    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams | None,
        return_extra: Literal[True],
    ) -> List[List[ResponseOutput]]: ...

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams | None,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """
        Send a batched generation request to the vLLM server.

        Args:
            prompts: List of prompts to generate text from.
            sampling_params: Optional sampling parameters for the generation.
            return_extra: If True, return the full `ResponseOutput` objects instead
                of just the generated text.

        Returns:
            List of lists containing generated responses. If `return_extra` is True,
            returns a list of lists of ResponseOutput objects, otherwise just the text.
        """
        payload = GenerateRequest(
            prompts=prompts,
            params=sampling_params,
        )

        try:
            # Pre-encode with msgspec rather than letting ``requests`` call
            # ``json.dumps`` internally.
            response = self.session.post(
                url=f"{self.base_url}/generate",
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

    def close(self) -> None:
        """Close the underlying HTTP session. Never raises."""
        if self.session is None:
            return
        try:
            self.session.close()
            self.session = None
        except Exception:
            logger.warning("Failed to close session", exc_info=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()
