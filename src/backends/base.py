"""InferenceBackend ABC — every engine (llama.cpp, MLX, vLLM, ...) implements this."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.2
    stop: Sequence[str] = ()
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class TokenChunk:
    text: str
    done: bool = False
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    detail: str = ""


class InferenceBackend(ABC):
    """Common surface for all inference engines. Implementations must be async."""

    backend_id: str

    @abstractmethod
    async def generate_stream(self, req: GenerateRequest) -> AsyncIterator[TokenChunk]:
        """Yield TokenChunks; final chunk must have done=True."""

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text. Batched."""

    @abstractmethod
    async def health(self) -> HealthStatus:
        """Cheap liveness probe."""

    async def close(self) -> None:
        """Optional teardown hook."""
        return None
