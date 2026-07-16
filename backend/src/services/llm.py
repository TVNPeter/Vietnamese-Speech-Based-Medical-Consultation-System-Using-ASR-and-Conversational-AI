"""LLM service: manages llama-server subprocess and API calls."""

import asyncio
import json
import logging
import subprocess
from collections.abc import AsyncIterator

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Manages a llama-server subprocess and streams chat completions."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._base_url = (
            f"http://{settings.LLAMA_SERVER_HOST}:{settings.LLAMA_SERVER_PORT}"
        )
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=120.0)

    async def start(self) -> None:
        """Spawn the llama-server process and wait until it is healthy."""
        server_path = settings.resolve(settings.LLAMA_SERVER_PATH)
        model_path = settings.resolve(settings.LLM_MODEL_PATH)

        if not server_path.is_file():
            raise FileNotFoundError(f"llama-server not found: {server_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"LLM model not found: {model_path}")
        template_path = settings.resolve(settings.LLAMA_CHAT_TEMPLATE_FILE)
        if not template_path.is_file():
            raise FileNotFoundError(f"LLM chat template not found: {template_path}")

        cmd = [
            str(server_path),
            "--model",
            str(model_path),
            "--host",
            settings.LLAMA_SERVER_HOST,
            "--port",
            str(settings.LLAMA_SERVER_PORT),
            "--ctx-size",
            str(settings.LLAMA_CTX_SIZE),
            "--n-gpu-layers",
            str(settings.LLAMA_GPU_LAYERS),
            "--chat-template-file",
            str(template_path),
            "--reasoning",
            settings.LLAMA_REASONING,
            "--reasoning-budget",
            str(settings.LLAMA_REASONING_BUDGET),
        ]
        logger.info("Starting llama-server: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await self._wait_for_health()
        logger.info("llama-server is healthy at %s", self._base_url)

    async def _wait_for_health(self, timeout: float = 120.0) -> None:
        """Poll the /health endpoint until the server is ready."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if self._process and self._process.poll() is not None:
                stderr_output = ""
                if self._process.stderr:
                    stderr_output = self._process.stderr.read().decode()
                raise RuntimeError(
                    f"llama-server exited with code "
                    f"{self._process.returncode}: {stderr_output[:500]}"
                )
            try:
                resp = await self._client.get("/health")
                if resp.status_code == 200:
                    return
            except httpx.ConnectError:
                pass
            await asyncio.sleep(1.0)
        raise TimeoutError(f"llama-server did not become healthy within {timeout}s")

    async def stop(self) -> None:
        """Terminate the llama-server process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=10)
            logger.info("llama-server terminated.")
        await self._client.aclose()

    async def generate_stream(
        self,
        user_message: str,
        system_prompt: str,
    ) -> AsyncIterator[str]:
        """Stream tokens from llama-server's chat completions endpoint."""
        payload = {
            "model": "local",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": True,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        }
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            timeout=300.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0]["delta"]
                if delta.get("content"):
                    yield delta["content"]
