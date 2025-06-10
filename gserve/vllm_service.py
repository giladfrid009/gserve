import os
import pathlib
import sys
import socket
import time
import subprocess
import atexit
import json
from typing import Dict, List, Optional, TypeVar, Sequence
from concurrent.futures import ThreadPoolExecutor

import requests
from vllm import SamplingParams

from gserve.vllm_client import VLLMClient
from gserve.schema import ResponseOutput
from gserve.configs import LLMConfig, ServeConfig
from gserve.log_utils import LOG_LEVEL_ENV

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

        self._process: Optional[subprocess.Popen] = None

        # Ensure cleanup at interpreter exit
        atexit.register(self.shutdown)

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
            python gserve/vllm_server.py --serve --model <model_name>
                --host <host> --port <port> --gpus <comma-joined gpu_ids>
                --llm_kwargs <json-encoded dict of additional args>

        Then poll GET /health until HTTP 200 or timeout. Raises RuntimeError on failure.
        """
        if self._process is not None:
            raise RuntimeError("vLLM server is already running.")

        # 1) Build subprocess env: inherit but override CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)
        # Propagate logging level to the subprocess so it uses the same
        # verbosity and formatting.
        env[LOG_LEVEL_ENV] = os.environ.get(LOG_LEVEL_ENV, "INFO")

        # Ensure PYTHONPATH includes the root of this package.
        # This allows importing local modules from the script correctly.
        root_path = str(pathlib.Path(__file__).resolve().parent.parent.absolute())
        script_path = str(pathlib.Path(root_path, "gserve", "vllm_server.py").absolute())
        env["PYTHONPATH"] = root_path + ((":" + env["PYTHONPATH"]) if "PYTHONPATH" in env else "")

        # 2) Build the command
        cmd = [
            sys.executable,
            script_path,
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

        logger.info("Launching subprocess:\n    %s", " ".join(cmd))

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=None if self.verbose else subprocess.PIPE,
                stderr=None if self.verbose else subprocess.PIPE,
                env=env,
                text=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to launch subprocess: {e!r}")

        # Poll GET /health until 200 or timeout
        health_url = f"http://{self.host}:{self.port}/health"
        t0 = time.time()
        while True:
            # 1) If subprocess died, capture logs and error
            if self._process.poll() is not None:
                out, err = self.fetch_logs()
                if out:
                    logger.error("Subprocess STDOUT:\n%s", out)
                if err:
                    logger.error("Subprocess STDERR:\n%s", err)
                raise RuntimeError("Subprocess terminated prematurely.")
            # 2) Try health endpoint
            try:
                resp = requests.get(health_url, timeout=1.0)
                if resp.status_code == 200:
                    logger.info("Server is healthy at %s", health_url)
                    break
            except Exception:
                pass

            # 3) Timeout?
            if self.startup_timeout is not None and time.time() - t0 > self.startup_timeout:
                self._terminate_process()
                out, err = self.fetch_logs()
                if out:
                    logger.error("Subprocess STDOUT:\n%s", out)
                if err:
                    logger.error("Subprocess STDERR:\n%s", err)
                raise RuntimeError(
                    f"Timeout ({self.startup_timeout}s) waiting for health check."
                )
            time.sleep(0.1)

    def is_running(self) -> bool:
        """
        Return True if the subprocess exists and is still alive.
        """
        return self._process is not None and (self._process.poll() is None)

    def shutdown(self) -> None:
        """Gracefully terminate the server subprocess. Idempotent. Never raises."""
        if self._process is None:
            return

        logger.info("Shutting down subprocess...")
        try:
            resp = requests.post(f"http://{self.host}:{self.port}/shutdown", timeout=5)
            if resp.status_code != 200:
                logger.warning("Shutdown request returned HTTP %s", resp.status_code)
        except requests.RequestException as e:
            logger.warning("Shutdown request failed: %s", e)

        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown timed out; forcing kill.")
            self._terminate_process()
        finally:
            self._process = None

        self._process = None

    def _terminate_process(self) -> None:
        """
        Helper: send SIGTERM, wait 5s, then SIGKILL if still alive. Never raises.
        """
        if self._process is None:
            return
        try:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Did not exit after SIGTERM; forcing kill.")
                self._process.kill()
                self._process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error terminating subprocess: {e!r}")
        finally:
            self._process = None

    def fetch_logs(self) -> tuple[str, str]:
        """Return the subprocess STDOUT/STDERR if available."""
        if self._process is None:
            return "", ""
        try:
            out, err = self._process.communicate(timeout=1)
        except Exception:
            out, err = "", ""
        return out, err

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()


class VLLMService:
    """
    Convenience wrapper bundling :class:`VLLMServer` and :class:`VLLMClient`.
    Typical usage::

        service = VLLMService(
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

    def _split_batches(self, items: Sequence[T]) -> List[List[T]]:
        """Split a list of items into batches, one for each server."""
        num_servers = len(self.servers)
        if num_servers == 0:
            raise RuntimeError("No servers available. Call .start() first.")

        total = len(items)
        base = total // num_servers
        extras = total % num_servers

        batches: List[List[T]] = []
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
                            "Failed to start server on %s:%s: %s", srv.host, srv.port, e
                        )
                        out, err = srv.fetch_logs()
                        if out:
                            logger.error("Subprocess STDOUT:\n%s", out)
                        if err:
                            logger.error("Subprocess STDERR:\n%s", err)
                        raise

            for srv in started:
                client = VLLMClient(
                    host=srv.host,
                    port=srv.port,
                    timeout=self._serve_config.client_timeout,
                )
                self.clients.append(client)

            self.servers = started
            logger.info(
                "Started %d server(s) listening on %s",
                len(self.servers),
                ", ".join(f"{srv.host}:{srv.port}" for srv in self.servers),
            )
        except Exception:
            for srv in started:
                srv.shutdown()
            for client in self.clients:
                client.close()
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

        if len(conversations) == 0:
            return []

        batches = self._split_batches(conversations)
        results: List[List] = []
        with ThreadPoolExecutor(max_workers=len(self.clients)) as ex:
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

        if len(prompts) == 0:
            return []

        batches = self._split_batches(prompts)
        results: List[List] = []
        with ThreadPoolExecutor(max_workers=len(self.clients)) as ex:
            futures = [ex.submit(client.generate, batch, sampling_params, return_extra) for client, batch in zip(self.clients, batches)]
            for fut in futures:
                results.append(fut.result())

        merged: List[List] = []
        for res in results:
            merged.extend(res)
        return merged

    def shutdown(self) -> None:
        """Shut down all servers and close clients. Idempotent. Never raises."""
        servers = self.servers[:]
        clients = self.clients[:]

        self.servers.clear()
        self.clients.clear()

        if not servers and not clients:
            return

        logger.info("Shutting down %d server(s)...", len(servers))

        for client in clients:
            client.close()

        if not servers:
            return

        with ThreadPoolExecutor(max_workers=len(servers)) as ex:
            futures = [ex.submit(s.shutdown) for s in servers]
            for fut in futures:
                fut.result()

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
