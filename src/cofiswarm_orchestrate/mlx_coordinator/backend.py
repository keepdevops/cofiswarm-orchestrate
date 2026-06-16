"""MlxBackend — InferenceBackend implementation for mlx_lm.server endpoints.

Key MLX-specific behaviours (mirrors cpp_core/src/agent_client.cpp MLX path):
  - system_prompt is merged into the user turn as a single message (no system role)
  - health probe uses GET /v1/models, not /health
  - inflight counter is maintained per-port for pressure reporting
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Sequence

import aiohttp

from backends.base import GenerateRequest, HealthStatus, InferenceBackend, TokenChunk

logger = logging.getLogger(__name__)

# Per-port inflight counters and serialization semaphores.
_inflight: dict[int, int] = {}
_inflight_lock = asyncio.Lock()
_port_semaphores: dict[int, asyncio.Semaphore] = {}
_semaphores_lock = asyncio.Lock()


async def _get_semaphore(port: int) -> asyncio.Semaphore:
    async with _semaphores_lock:
        if port not in _port_semaphores:
            _port_semaphores[port] = asyncio.Semaphore(1)
        return _port_semaphores[port]


async def _inc(port: int) -> None:
    async with _inflight_lock:
        _inflight[port] = _inflight.get(port, 0) + 1


async def _dec(port: int) -> None:
    async with _inflight_lock:
        _inflight[port] = max(0, _inflight.get(port, 0) - 1)


def get_pressure() -> dict[int, int]:
    return dict(_inflight)


class MlxBackend(InferenceBackend):
    """Streams from one mlx_lm.server instance (OpenAI-compatible)."""

    backend_id = "mlx"

    def __init__(self, port: int, agent_id: str, system_prompt: str = "",
                 max_tokens: int = 512, temperature: float = 0.2) -> None:
        self.port = port
        self.agent_id = agent_id
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300, connect=5),
            )
        return self._session

    def _build_messages(self, prompt: str) -> list[dict]:
        """Merge system_prompt + prompt into a single user message (MLX convention)."""
        if self.system_prompt:
            content = f"{self.system_prompt}\n\n{prompt}"
        else:
            content = prompt
        return [{"role": "user", "content": content}]

    async def generate_stream(self, req: GenerateRequest) -> AsyncIterator[TokenChunk]:
        messages = self._build_messages(req.prompt)
        # Omit the OpenAI "model" field: mlx_lm.server serves the single model it
        # was launched with and returns 404 for an unknown id like "default". The
        # C++ MLX path (agent_stream_llama.h stream_mlx) omits it for the same reason.
        payload = {
            "messages": messages,
            "max_tokens": req.max_tokens or self.max_tokens,
            "temperature": req.temperature or self.temperature,
            "stream": True,
        }
        if req.stop:
            payload["stop"] = list(req.stop)

        sem = await _get_semaphore(self.port)
        async with sem:
            async for chunk in self._do_generate(req, messages, payload):
                yield chunk

    async def _do_generate(self, req: GenerateRequest, messages: list, payload: dict) -> AsyncIterator[TokenChunk]:
        await _inc(self.port)
        t_start = time.monotonic()
        completion_tokens = 0

        try:
            session = self._get_session()
            async with session.post(
                f"{self.base_url}/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("mlx-backend %s: HTTP %d: %s", self.agent_id, resp.status, body)
                    yield TokenChunk(text=f"[mlx error {resp.status}]", done=True)
                    return

                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[5:].strip()
                    if payload_str == "[DONE]":
                        break
                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError as exc:
                        logger.error("mlx-backend %s: bad SSE JSON: %s", self.agent_id, exc)
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        completion_tokens += 1
                        yield TokenChunk(text=text)

        except asyncio.TimeoutError:
            logger.error("mlx-backend %s: request timed out", self.agent_id)
            yield TokenChunk(text="[timeout]", done=True)
            return
        except aiohttp.ClientError as exc:
            logger.error("mlx-backend %s: connection error: %s", self.agent_id, exc)
            yield TokenChunk(text=f"[connection error: {exc}]", done=True)
            return
        finally:
            await _dec(self.port)
            elapsed = time.monotonic() - t_start
            logger.debug("mlx-backend %s: %d tokens in %.2fs", self.agent_id, completion_tokens, elapsed)

        yield TokenChunk(text="", done=True)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        logger.error("mlx-backend %s: embed() not supported — use RAG sidecar", self.agent_id)
        raise NotImplementedError("MLX coordinator does not serve embeddings directly")

    async def health(self) -> HealthStatus:
        """Probe via GET /v1/models — mlx_lm.server exposes this, not /health."""
        try:
            session = self._get_session()
            async with session.get(f"{self.base_url}/models", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return HealthStatus(ok=True, detail=f"port {self.port} ok")
                return HealthStatus(ok=False, detail=f"port {self.port} HTTP {resp.status}")
        except Exception as exc:
            logger.error("mlx-backend %s: health check failed: %s", self.agent_id, exc)
            return HealthStatus(ok=False, detail=str(exc))

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
