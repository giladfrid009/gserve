import os
import sys
import socket
import time
import subprocess
import atexit
import json
import asyncio
from typing import Dict, List, Optional, TypeVar, Sequence
from concurrent.futures import ThreadPoolExecutor

import requests
from vllm import SamplingParams

from src.vllm_client import VLLMClient
from src.vllm_server import ResponseOutput
from src.configs import LLMConfig, ServeConfig

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMServer:
    """
    Spawns a subprocess running ``python vllm_server.py --serve ...``.
    The subprocess hosts a FastAPI server that loads vLLM on specified GPUs and
    exposes ``/health``, ``/chat`` and ``/generate`` endpoints.

    Usage:

        server = VLLMServer(
            llm_config=LLMConfig(model_name="meta-llama/Llama-3.2-1b-Instruct"),
            gpu_ids=[0],
        )
        server.start()
        # Now /health, /chat, /generate are available at server.host:server.port
        server.shutdown()
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        gpu_ids: List[int],
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        startup_timeout: Optional[float] = 15.0,
        server_script_path: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Args:
            llm_config: Configuration for the LLM instance.
            gpu_ids: GPUs visible to this server process.
            host: Host/IP for the server (default "127.0.0.1").
            port: If None, auto-pick a free port; otherwise bind exactly to this port.
            startup_timeout: Seconds to wait for GET /health to return 200.
                ``None`` disables the timeout.
            server_script_path: Path to vllm_server.py. If None, assume same directory.
            verbose: If True, forward stdout/stderr from the server subprocess
                to the parent process.
        """
        self.llm_config = llm_config
        self.gpu_ids = gpu_ids.copy()
        self.host = host
        self.startup_timeout = startup_timeout
        self.verbose = verbose

        # Determine port
        if port is None:
            self.port = self._find_free_port()
        else:
            self.port = port

        # Locate vllm_server.py
        if server_script_path is None:
            this_dir = os.path.dirname(os.path.realpath(__file__))
            self.server_script = os.path.join(this_dir, "vllm_server.py")
        else:
            self.server_script = server_script_path

        if not os.path.isfile(self.server_script):
            raise FileNotFoundError(f"Cannot find server script at: {self.server_script}")

        self._process: Optional[subprocess.Popen] = None
        self._is_shut_down = False

        # Ensure cleanup at interpreter exit
        atexit.register(self._atexit_shutdown)

    @staticmethod
    def _find_free_port() -> int:
        """Ask the OS for a free TCP port (bind to port=0)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        return port

    def start(self) -> None:
        """
        Launch a subprocess:
            python vllm_server.py --serve --model <model_name>
                --host <host> --port <port> --gpus <comma-joined gpu_ids>
                --llm_kwargs <json-encoded dict of additional args>

        Then poll GET /health until HTTP 200 or timeout. Raises RuntimeError on failure.
        """
        if self._process is not None:
            raise RuntimeError("vLLM server is already running.")

        # 1) Build subprocess env: inherit but override CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)

        # 2) Build the command
        cmd = [
            sys.executable,
            self.server_script,
            "--serve",
            "--model",
            self.llm_config.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--gpus",
            ",".join(str(g) for g in self.gpu_ids),
        ]

        llm_args = {
            "dtype": self.llm_config.dtype,
            "tokenizer": self.llm_config.tokenizer,
            "tokenizer_mode": self.llm_config.tokenizer_mode,
            "trust_remote_code": self.llm_config.trust_remote_code,
            "max_model_len": self.llm_config.max_model_len,
            "download_dir": self.llm_config.download_dir,
            "quantization": self.llm_config.quantization,
            "revision": self.llm_config.revision,
            "tokenizer_revision": self.llm_config.tokenizer_revision,
            "seed": self.llm_config.seed,
            "gpu_memory_utilization": self.llm_config.gpu_memory_utilization,
            "enforce_eager": self.llm_config.enforce_eager,
            **self.llm_config.llm_kwargs,
        }

        llm_args = {k: v for k, v in llm_args.items() if v is not None}
        if llm_args:
            cmd.extend(["--llm_kwargs", json.dumps(llm_args)])

        if self.llm_config.lora_path is not None:
            cmd.extend(["--lora_path", self.llm_config.lora_path])

        logger.info(f"[VLLMServer] Launching subprocess:\n    {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=None if self.verbose else subprocess.PIPE,
                stderr=None if self.verbose else subprocess.PIPE,
                env=env,
                text=True,
            )
        except Exception as e:
            raise RuntimeError(f"[VLLMServer] Failed to launch subprocess: {e!r}")

        # Poll GET /health until 200 or timeout
        health_url = f"http://{self.host}:{self.port}/health"
        t0 = time.time()
        while True:
            # 1) If subprocess died, capture logs and error
            if self._process.poll() is not None:
                out, err = self.fetch_logs()
                if out:
                    logger.error("[VLLMServer STDOUT]\n%s", out)
                if err:
                    logger.error("[VLLMServer STDERR]\n%s", err)
                raise RuntimeError("[VLLMServer] Subprocess terminated prematurely.")
            # 2) Try health endpoint
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code == 200:
                    logger.info(f"[VLLMServer] Server is healthy at {health_url}")
                    break
            except Exception:
                pass

            # 3) Timeout?
            if self.startup_timeout is not None and time.time() - t0 > self.startup_timeout:
                self._terminate_process()
                out, err = self.fetch_logs()
                if out:
                    logger.error("[VLLMServer STDOUT]\n%s", out)
                if err:
                    logger.error("[VLLMServer STDERR]\n%s", err)
                raise RuntimeError(f"[VLLMServer] Timeout ({self.startup_timeout}s) waiting for health check.")
            time.sleep(0.1)

    def is_running(self) -> bool:
        """
        Return True if the subprocess exists and is still alive.
        """
        return self._process is not None and (self._process.poll() is None)

    def shutdown(self) -> None:
        """
        Gracefully terminate the server subprocess. Idempotent.
        """
        if self._is_shut_down:
            return
        self._is_shut_down = True

        if self._process is None:
            return

        logger.info("[VLLMServer] Shutting down subprocess...")
        try:
            resp = requests.post(f"http://{self.host}:{self.port}/shutdown", timeout=5)
            if resp.status_code != 200:
                logger.warning("[VLLMServer] Shutdown request returned HTTP %s", resp.status_code)
        except requests.RequestException as e:
            logger.warning("[VLLMServer] Shutdown request failed: %s", e)

        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("[VLLMServer] Graceful shutdown timed out; forcing kill.")
            self._terminate_process()

        self._process = None

    def _terminate_process(self) -> None:
        """
        Helper: send SIGTERM, wait 5s, then SIGKILL if still alive.
        """
        if self._process is None:
            return
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("[VLLMServer] Did not exit after SIGTERM; forcing kill.")
                self._process.kill()
                self._process.wait(timeout=5)
        except Exception as e:
            logger.error(f"[VLLMServer] Error terminating subprocess: {e!r}")

    def fetch_logs(self) -> tuple[str, str]:
        """Return the subprocess STDOUT/STDERR if available."""
        if self._process is None:
            return "", ""
        try:
            out, err = self._process.communicate(timeout=1)
        except Exception:
            out, err = "", ""
        return out, err

    def _atexit_shutdown(self) -> None:
        """
        Called automatically at interpreter exit to ensure cleanup.
        """
        try:
            self.shutdown()
        except Exception:
            pass

    def __del__(self):
        # Fallback in case shutdown wasn’t explicitly called
        self._atexit_shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()


class VLLMService:
    """
    Convenience wrapper bundling VLLMServer + VLLMClient. You only need to interact
    with this one class:

        service = VLLMBatchedService(
            model_name="meta-llama/Llama-3.2-1b-Instruct",
            gpu_ids=[0,1],
            host="127.0.0.1",
            port=None,
            verbose=True,
        )
        service.start()
        chat_answers = service.chat([...])         # batched chat
        gen_answers  = service.generate([...])     # batched generate
        service.shutdown()

    The server subprocess is spawned in `start()`, and the client is automatically created.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        serve_config: ServeConfig,
    ):
        """
        Args:
            llm_config: Configuration of the model to serve.
            serve_config: Parameters controlling how the service runs.
        """

        self._llm_config = llm_config
        self._serve_config = serve_config

        self._gpu_ids = serve_config.gpu_ids

        self.servers: List[VLLMServer] = []
        self.clients: List[VLLMClient] = []

    T = TypeVar("_T")

    @staticmethod
    def _split_batches(items: Sequence[T], num_servers: int) -> List[List[T]]:
        """Split ``items`` into ``num_servers`` batches preserving order."""
        total = len(items)
        base = total // num_servers
        extras = total % num_servers

        batches: List[List[VLLMService.T]] = []
        start = 0
        for i in range(num_servers):
            size = base + (1 if i < extras else 0)
            batches.append(list(items[start : start + size]))
            start += size
        return batches

    def start(self) -> None:
        """Start all server subprocesses concurrently and create clients."""
        if self.servers:
            raise RuntimeError("Service already started")

        servers: List[VLLMServer] = []
        port_counter = self._serve_config.port
        for gpu_id in self._gpu_ids:
            srv = VLLMServer(
                llm_config=self._llm_config,
                gpu_ids=[gpu_id],
                host=self._serve_config.host,
                port=port_counter,
                startup_timeout=self._serve_config.startup_timeout,
                server_script_path=self._serve_config.server_script_path,
                verbose=self._serve_config.verbose,
            )
            servers.append(srv)
            if port_counter is not None:
                port_counter += 1

        started: List[VLLMServer] = []
        try:
            with ThreadPoolExecutor(max_workers=len(servers)) as ex:
                futures = {ex.submit(s.start): s for s in servers}
                for fut, srv in futures.items():
                    try:
                        fut.result()
                        started.append(srv)
                    except Exception as e:
                        logger.error(
                            "[VLLMService] Failed to start server on %s:%s: %s",
                            srv.host,
                            srv.port,
                            e,
                        )
                        out, err = srv.fetch_logs()
                        if out:
                            logger.error("[VLLMServer STDOUT]\n%s", out)
                        if err:
                            logger.error("[VLLMServer STDERR]\n%s", err)
                        raise

            for srv in started:
                client = VLLMClient(
                    host=srv.host,
                    port=srv.port,
                    timeout=self._serve_config.client_timeout,
                )
                self.clients.append(client)

            self.servers = started
        except Exception:
            for srv in started:
                try:
                    srv.shutdown()
                except Exception:
                    pass
            raise

    def is_running(self) -> bool:
        """Return True if at least one server subprocess is still alive."""
        return any(server.is_running() for server in self.servers)

    def chat(
        self,
        conversations: List[List[Dict[str, str]]],
        sampling_params: SamplingParams | None = None,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """
        Batched chat.
        Returns a list of lists, each sub-list corresponds to number of outputs (n).
        """
        if not self.clients:
            raise RuntimeError("Service not started. Call .start() first.")
        if sampling_params is None:
            sampling_params = SamplingParams()

        num_servers = len(self.clients)
        if len(conversations) == 0:
            return []

        batches = self._split_batches(conversations, num_servers)
        results: List[List] = []
        with ThreadPoolExecutor(max_workers=num_servers) as ex:
            futures = [ex.submit(client.chat, batch, sampling_params, return_extra) for client, batch in zip(self.clients, batches)]
            for fut in futures:
                results.append(fut.result())

        merged: List[List] = []
        for res in results:
            merged.extend(res)
        return merged

    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams | None = None,
        return_extra: bool = False,
    ) -> List[List[str]] | List[List[ResponseOutput]]:
        """
        Batched generate.
        Returns a list of lists, each sub-list corresponds to number of outputs (n).
        Raises if not started.
        """
        if not self.clients:
            raise RuntimeError("Service not started. Call .start() first.")
        if sampling_params is None:
            sampling_params = SamplingParams()

        num_servers = len(self.clients)
        if len(prompts) == 0:
            return []

        batches = self._split_batches(prompts, num_servers)
        results: List[List] = []
        with ThreadPoolExecutor(max_workers=num_servers) as ex:
            futures = [ex.submit(client.generate, batch, sampling_params, return_extra) for client, batch in zip(self.clients, batches)]
            for fut in futures:
                results.append(fut.result())

        merged: List[List] = []
        for res in results:
            merged.extend(res)
        return merged

    def shutdown(self) -> None:
        """Shut down all servers concurrently. Idempotent."""
        servers = self.servers[:]
        self.servers.clear()
        self.clients.clear()

        if not servers:
            return

        with ThreadPoolExecutor(max_workers=len(servers)) as ex:
            futures = [ex.submit(s.shutdown) for s in servers]
            for fut in futures:
                try:
                    fut.result()
                except Exception as e:
                    logger.error("[VLLMService] Error shutting down server: %s", e)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
