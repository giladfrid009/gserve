from typing import List, Dict, Any, Optional
import msgspec
from vllm import SamplingParams
from vllm.sequence import SampleLogprobs


class ChatRequest(msgspec.Struct, array_like=True, omit_defaults=True, forbid_unknown_fields=True):
    """Payload for batched chat requests."""

    conversations: List[List[Dict[str, Any]]]
    params: Optional[SamplingParams] = None


class GenerateRequest(msgspec.Struct, array_like=True, omit_defaults=True, forbid_unknown_fields=True):
    """Payload for batched generation requests."""

    prompts: List[str]
    params: Optional[SamplingParams] = None


class ResponseOutput(msgspec.Struct, array_like=True, omit_defaults=True, forbid_unknown_fields=True):
    """Single generation response from vLLM."""

    text: str
    logprobs: Optional[SampleLogprobs] = None
