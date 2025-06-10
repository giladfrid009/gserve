# gserve: Simplified vLLM Serving

`gserve` is a Python library designed to simplify the deployment and interaction with [vLLM](https://github.com/vllm-project/vllm) for large language model (LLM) inference. It provides a high-level interface to spawn vLLM-powered FastAPI servers as subprocesses, manage their lifecycle, and send batched requests for chat completions and text generation.

## Overview

The core idea behind `gserve` is to make it easy to:
1.  **Launch vLLM Instances**: Programmatically start one or more vLLM servers, each potentially on a dedicated GPU.
2.  **Configure Models**: Easily specify the model, LoRA adapters, quantization, and other vLLM parameters.
3.  **Interact via Client**: Use a simple client to send batched requests to the server(s) for efficient inference.
4.  **Manage Resources**: Handle server startup, shutdown, and health checks automatically.

`gserve` is particularly useful for integrating vLLM inference into larger Python applications, research workflows, or for building custom LLM-based services.

## Features

*   **Subprocess Management**: Spawns and manages vLLM FastAPI servers in separate processes.
*   **Automatic Port Allocation**: Finds free ports for servers if not specified.
*   **Configuration Classes**: Uses `LLMConfig` and `ServeConfig` for clear and structured server/model setup.
*   **Batched Inference**: Supports batched requests for both `/chat` and `/generate` endpoints for higher throughput.
*   **Multi-GPU Support**: Easily distribute inference across multiple GPUs by launching multiple server instances, each pinned to specific GPU(s).
*   **LoRA Support**: Load and use LoRA adapters with your models.
*   **Health Checks & Timeouts**: Includes startup health checks and configurable timeouts.
*   **Graceful Shutdown**: Provides mechanisms for gracefully shutting down server subprocesses.
*   **Structured Logging**: Configurable logging with propagation to subprocesses.
*   **Asynchronous Operations**: Uses `ThreadPoolExecutor` for non-blocking server startup and request distribution.
*   **Context Manager Support**: `VLLMServer` and `VLLMService` can be used as context managers for automatic resource cleanup.

## Installation

Python 3.12 or newer is required. Install the locked dependencies using [uv](https://github.com/astral-sh/uv) or plain `pip`:

```bash
uv pip install -r uv.lock  # or: pip install -r uv.lock
```

To work on the library itself, install it in editable mode:

```bash
pip install -e .
```

## Quick Start

The primary way to use `gserve` is through the `VLLMService` class, which bundles server management and client interaction.

### Using `VLLMService` (Recommended)

This example demonstrates how to start a service for a Llama model on GPU 0, send chat and generation requests, and then shut it down.

```python
from gserve import VLLMService, LLMConfig, ServeConfig
from vllm import SamplingParams

# 1. Configure the LLM and Serving parameters
llm_config = LLMConfig(
    model_name="meta-llama/Llama-3.2-8B-Instruct", # Or any other model
    # tokenizer="meta-llama/Llama-3.2-8B-Instruct", # Optional: if different from model_name
    # trust_remote_code=True, # If model requires it
    # dtype="bfloat16",
    # quantization="awq", # Example: if using a quantized model
    # lora_path="/path/to/your/lora_adapter", # Optional: if using LoRA
)

serve_config = ServeConfig(
    gpu_ids=[0],  # Use GPU 0. For multiple GPUs, e.g., [0, 1]
    # port=8000, # Optional: specify a starting port, otherwise auto-assigned
    startup_timeout=300.0, # Increase if your model takes longer to load
    verbose=True # Show server logs
)

# 2. Initialize and start the service
# This will spawn a vLLM server subprocess
service = VLLMService(llm_config=llm_config, serve_config=serve_config)

# Use as a context manager for automatic shutdown
with service:
    print(f"Service started. Servers running: {service.is_running()}")

    # 3. Prepare requests
    conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        [
            {"role": "user", "content": "Explain the theory of relativity in simple terms."}
        ]
    ]
    prompts = [
        "The quick brown fox jumps over the lazy ",
        "Once upon a time, in a land far, far away, "
    ]

    # Optional: Define sampling parameters
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=100)

    # 4. Send requests
    # Chat completions
    chat_responses = service.chat(conversations, sampling_params=sampling_params)
    for i, response_list in enumerate(chat_responses):
        print(f"Chat Response for conversation {i+1}:")
        for output in response_list:
            print(f"- {output}") # By default, returns list of strings

    # Text generation
    # To get full ResponseOutput (including logprobs if enabled in SamplingParams)
    generate_responses_extra = service.generate(prompts, sampling_params=sampling_params, return_extra=True)
    for i, response_list in enumerate(generate_responses_extra):
        print(f"Generation Response for prompt {i+1}:")
        for output_obj in response_list:
            print(f"- Text: {output_obj.text}")
            # print(f"- Logprobs: {output_obj.logprobs}") # If logprobs were requested

# Service is automatically shut down when exiting the 'with' block
print(f"Service shut down. Servers running: {service.is_running()}")

# Alternatively, without context manager:
# service.start()
# ... do work ...
# service.shutdown()
```

### Using `VLLMServer` and `VLLMClient` (Advanced)

For more fine-grained control, you can manage `VLLMServer` and `VLLMClient` instances separately.

```python
from gserve import VLLMServer, VLLMClient, LLMConfig
from vllm import SamplingParams

llm_config = LLMConfig(model_name="meta-llama/Llama-3.2-8B-Instruct")

# Start a single server
server = VLLMServer(
    llm_config=llm_config,
    gpu_ids=[0],
    verbose=True
)

with server: # Manages server.start() and server.shutdown()
    print(f"Server started on {server.host}:{server.port}")

    # Create a client for the server
    with VLLMClient(host=server.host, port=server.port) as client:
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(max_tokens=50)
        
        responses = client.generate(prompts, sampling_params)
        print(responses[0][0])

print("Server shut down.")
```

## Configuration

Configuration is handled by two dataclasses: `LLMConfig` and `ServeConfig`.

### `LLMConfig`

Defines parameters for the `vllm.LLM` instance.

```python
@dataclass
class LLMConfig:
    model_name: str  # HuggingFace model ID or local path
    tokenizer: Optional[str] = None  # Path to tokenizer, if different from model
    tokenizer_mode: str = "auto"  # "auto" or "slow"
    trust_remote_code: bool = False # Allow remote code for model
    dtype: str = "bfloat16"  # Data type like "float16", "bfloat16", "float32"
    quantization: Optional[str] = None  # E.g., "awq", "gptq"
    revision: Optional[str] = None  # Model revision
    tokenizer_revision: Optional[str] = None # Tokenizer revision
    seed: int = 0 # Random seed for reproducibility
    gpu_memory_utilization: Optional[float] = None # Between 0 and 1
    enforce_eager: bool = False # Force eager execution
    max_model_len: Optional[int] = None # Maximum model context length
    download_dir: Optional[str] = None # Directory to download models
    lora_path: Optional[str] = None # Path to LoRA adapter
    llm_kwargs: Dict[str, Any] = field(default_factory=dict) # Additional kwargs for vllm.LLM
```

### `ServeConfig`

Defines parameters for how the `VLLMService` or `VLLMServer` runs.

```python
@dataclass
class ServeConfig:
    gpu_ids: List[int] # List of GPU IDs to use. Each ID can spawn a server instance.
    host: str = "127.0.0.1" # Host for the server(s)
    port: Optional[int] = None # Starting port. If None, auto-assigned. Incremented for multiple servers.
    startup_timeout: Optional[float] = 180.0 # Seconds to wait for server health check
    client_timeout: Optional[float] = 30.0 # HTTP client timeout for requests
    verbose: bool = False # Forward server subprocess stdout/stderr
```

## API Reference

### `VLLMService`
*   `__init__(llm_config: LLMConfig, serve_config: ServeConfig)`: Initializes the service.
*   `start()`: Starts all configured vLLM server subprocesses and their clients.
*   `shutdown()`: Gracefully shuts down all servers and closes clients.
*   `is_running() -> bool`: Checks if at least one server is active.
*   `chat(conversations, sampling_params, return_extra) -> List[List[str]] | List[List[ResponseOutput]]`: Sends batched chat requests.
*   `generate(prompts, sampling_params, return_extra) -> List[List[str]] | List[List[ResponseOutput]]`: Sends batched generation requests.
*   Context manager support (`__enter__`, `__exit__`) for automatic start/shutdown.

### `VLLMServer`
*   `__init__(llm_config, gpu_ids, host, port, startup_timeout, verbose)`: Initializes a single server instance.
*   `start()`: Launches the vLLM server subprocess.
*   `shutdown()`: Gracefully terminates the server subprocess.
*   `is_running() -> bool`: Checks if the subprocess is alive.
*   `host`, `port`: Network address of the server.
*   Context manager support.

### `VLLMClient`
*   `__init__(host, port, timeout)`: Initializes a client to connect to a specific server.
*   `chat(...)`: Sends a chat request to the connected server.
*   `generate(...)`: Sends a generation request to the connected server.
*   `close()`: Closes the HTTP session.
*   Context manager support.

## Server Endpoints

The `vllm_server.py` script, when run by `VLLMServer`, exposes the following FastAPI endpoints:

*   **GET `/health`**: Health check. Returns `{"status": "ok"}` if the server and LLM are ready.
*   **POST `/chat`**: Accepts a `ChatRequest` JSON payload.
    *   Payload: `{"conversations": [[{"role": "user", "content": "..."}], ...], "params": {...}}`
    *   `params` is an optional dictionary matching `vllm.SamplingParams` attributes.
    *   Returns a list of lists of `ResponseOutput` objects: `[[{"text": "...", "logprobs": ...}], ...]`
*   **POST `/generate`**: Accepts a `GenerateRequest` JSON payload.
    *   Payload: `{"prompts": ["prompt1", "prompt2", ...], "params": {...}}`
    *   `params` is an optional dictionary matching `vllm.SamplingParams` attributes.
    *   Returns a list of lists of `ResponseOutput` objects.
*   **POST `/shutdown`**: Initiates a graceful server shutdown.

The request/response schemas (`ChatRequest`, `GenerateRequest`, `ResponseOutput`) are defined in `gserve/schema.py` using `msgspec` for efficient serialization/deserialization.

## Logging

`gserve` uses Python's standard `logging` module.
*   Logging can be configured using `gserve.log_utils.setup_logging()`.
*   The log level is propagated to server subprocesses via the `GSERVE_LOG_LEVEL` environment variable.
*   By default, the log level is `INFO`. You can set the `GSERVE_LOG_LEVEL` environment variable (e.g., to `DEBUG`) before running your script to change verbosity.
    ```bash
    export GSERVE_LOG_LEVEL=DEBUG
    python your_script_using_gserve.py
    ```

## Advanced Topics

### Multi-GPU Serving

`VLLMService` can manage multiple vLLM server instances, each typically assigned to a different GPU. This is controlled by the `gpu_ids` parameter in `ServeConfig`.

```python
serve_config = ServeConfig(
    gpu_ids=[0, 1], # Will attempt to start two servers, one on GPU 0, one on GPU 1
    # port=8000, # Server on GPU 0 will use 8000, server on GPU 1 will use 8001
    verbose=True
)

# llm_config remains the same
llm_config = LLMConfig(model_name="meta-llama/Llama-3.2-8B-Instruct")

service = VLLMService(llm_config=llm_config, serve_config=serve_config)

with service:
    # Requests to service.chat() or service.generate() will be
    # automatically distributed among the available servers.
    prompts = [f"Prompt for server {i}" for i in range(10)]
    responses = service.generate(prompts)
    # ...
```
Each server subprocess will have `CUDA_VISIBLE_DEVICES` set appropriately.

### LoRA Adapters

To use a LoRA adapter, specify its path in `LLMConfig`:

```python
llm_config = LLMConfig(
    model_name="meta-llama/Llama-3.2-8B-Instruct",
    lora_path="/path/to/your/lora_weights_directory_or_file",
    # vLLM usually requires enable_lora=True in llm_kwargs if lora_path is set,
    # but gserve's vllm_server.py handles this automatically.
    # You can still pass it if needed for specific vLLM versions or setups:
    # llm_kwargs={"enable_lora": True}
)

# ... then use with VLLMService or VLLMServer as usual
```
The `vllm_server.py` script will initialize the LLM with `enable_lora=True` and create a `LoRARequest` if `lora_path` is provided.

## Development

### Running `vllm_server.py` Directly

You can run the FastAPI server script directly for testing or development purposes. This is what `VLLMServer` does under the hood.

```bash
# Ensure gserve is in your PYTHONPATH
export PYTHONPATH=/path/to/gserve_parent_directory:$PYTHONPATH
export GSERVE_LOG_LEVEL=DEBUG # Optional: for more verbose logging

python /path/to/gserve/gserve/vllm_server.py --serve \
    --model "meta-llama/Llama-3.2-8B-Instruct" \
    --host "127.0.0.1" \
    --port 8000 \
    --gpus "0" \
    # --llm_kwargs '{"quantization": "awq", "trust_remote_code": true}' \
    # --lora_path "/path/to/lora"

# Example with llm_kwargs:
# python gserve/vllm_server.py --serve \
#   --model "TheBloke/Llama-2-7b-Chat-AWQ" \
#   --gpus "0" \
#   --llm_kwargs '{"quantization": "awq", "dtype": "half", "max_model_len": 4096}'
```

**Arguments for `vllm_server.py --serve`:**
*   `--model`: (Required) Model name or path.
*   `--host`: Host IP to bind to (default: "127.0.0.1").
*   `--port`: Port to bind to (default: 8000).
*   `--gpus`: Comma-separated GPU indices for `CUDA_VISIBLE_DEVICES` (e.g., "0" or "0,1").
*   `--llm_kwargs`: JSON string of additional arguments for `vllm.LLM()`.
*   `--lora_path`: Path to LoRA adapter.

This allows you to test the server component independently of the `VLLMServer` wrapper.
