"""Configuration dataclasses for the inference/serving utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    """Configuration for the underlying vLLM model."""

    model_name: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    quantization: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    seed: int = 0
    gpu_memory_utilization: Optional[float] = None
    enforce_eager: bool = False
    max_model_len: Optional[int] = None
    download_dir: Optional[str] = None
    lora_path: Optional[str] = None
    llm_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServeConfig:
    """Configuration for :class:`~gserve.inference.vllm_service.VLLMService`."""

    gpu_ids: List[int]
    host: str = "127.0.0.1"
    port: Optional[int] = None
    startup_timeout: Optional[float] = 180.0
    client_timeout: Optional[float] = 30.0
    verbose: bool = False
