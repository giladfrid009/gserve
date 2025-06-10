# gserve

A minimal library for hosting [vLLM](https://github.com/vllm-project/vllm) models behind a small FastAPI server and a convenience client.  The project provides utilities to start a server in a subprocess, send batched chat or generation requests and cleanly shut everything down.

## Event loop behavior

The code exposes synchronous helper methods for ease of use.  Internally these methods rely on asynchronous implementations and use `asyncio.run()` to execute them.  Because `asyncio.run()` creates a new event loop, you **must not** call the synchronous methods (`start()`, `chat()`, `generate()`, `shutdown()` and others) while already inside another running event loop.  Doing so will raise a `RuntimeError` about a loop already running.  If you need to use the functionality from an async context, call the async variants directly instead of the sync wrappers.

In short:

- Non-async methods spin up a temporary event loop via `asyncio.run()` and delegate to async implementations.
- Calling them from inside an existing event loop will fail.

This design allows the library to be used with regular blocking code while still providing async implementations under the hood.

## Repository layout

```
gserve/
    configs.py       Dataclasses describing model and service configuration.
    vllm_server.py   FastAPI server exposing `/health`, `/chat`, `/generate` and `/shutdown` endpoints.
    vllm_client.py   Lightweight HTTP client for the server.
    vllm_service.py  Helper that manages server subprocesses and clients to provide batched chat/generate.
notebooks/
    test_vllm_service.ipynb  Example notebook using `VLLMService`.
tests/
```

### `configs.py`
Defines two dataclasses:
- `LLMConfig` – parameters for constructing the underlying `vllm.LLM` instance.
- `ServeConfig` – options controlling the server such as host, port and timeout values.

### `vllm_server.py`
Implements the FastAPI application.  On startup it loads `vllm.LLM` and exposes endpoints for batched chat and generation.  The `/shutdown` endpoint cleans up the model and stops the server process.

### `vllm_client.py`
Simple HTTP client built with `requests`.  It sends JSON payloads (encoded with `msgspec`) to the server and returns either plain text results or full objects containing logprob information.  It performs a quick health check upon creation.

### `vllm_service.py`
A higher level wrapper that starts one or more server subprocesses (one GPU each), creates corresponding clients and forwards chat/generate calls in batches.  Shutting down tears down clients and servers.  All public methods have synchronous signatures but use `asyncio.run()` internally to await subprocess start up and HTTP calls.

## Usage example

```python
from gserve.configs import LLMConfig, ServeConfig
from gserve.vllm_service import VLLMService

llm_cfg = LLMConfig(model_name="meta-llama/Llama-3.2-1b-Instruct")
serve_cfg = ServeConfig(gpu_ids=[0])

service = VLLMService(llm_cfg, serve_cfg)
service.start()  # spins up a subprocess running the FastAPI server

convs = [[{"role": "user", "content": "Hello"}]]
answers = service.chat(convs)
print(answers)

service.shutdown()
```

Remember to call the async APIs directly if running inside another event loop.

### Logging

`gserve` installs a default logging configuration on import if the
application has not set one.  Log entries follow the pattern:

```
INFO 06-10 15:23:31 [config.py:2118] Chunked prefill is enabled with max_num_batched_tokens=8192.
```

Call `gserve.setup_logging(level=logging.DEBUG)` to change the log level or
format at any time.  The chosen level is propagated to any subprocesses spawned
by :class:`gserve.VLLMService`.

## Installation

The project requires Python 3.12+.  Install dependencies with [`uv`](https://github.com/astral-sh/uv) or `pip`:

```bash
uv pip install -r uv.lock
```

To install an editable version of the package itself:

```bash
pip install -e .
```

## Notebook

The `notebooks/test_vllm_service.ipynb` notebook demonstrates how to start the service, send requests and shut it down.  Open it in JupyterLab to experiment.