"""OrchestrationMode ABC — every execution mode (flat, pipeline, ToT, ...) implements this.

A mode receives a ModeContext (the active swarm + backends) and a query, and
yields Events that the coordinator streams back to the client.

"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping

from backends.base import InferenceBackend
from cofiswarm_orchestrate.manager import AgentConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """One unit emitted by a mode (token chunk, agent boundary, final result)."""

    kind: str  # "token" | "agent_start" | "agent_end" | "result" | "error"
    agent_id: str | None = None
    text: str = ""
    meta: Mapping[str, Any] | None = None


@dataclass
class ModeContext:
    """Everything a mode needs to execute one query."""

    swarm: Mapping[str, AgentConfig]
    backends: Mapping[str, InferenceBackend]
    agents: list[str] = field(default_factory=list)  # subset of swarm participating
    params: dict[str, Any] = field(default_factory=dict)  # mode-specific knobs
    request_id: str = ""
    # Optional RAG plumbing. When both are set and an agent has rag.enabled
    # in its config, modes inject retrieved top-k chunks into the prompt.
    embedder: Any = None
    store: Any = None

    def agent(self, agent_id: str) -> AgentConfig:
        try:
            return self.swarm[agent_id]
        except KeyError as exc:
            logger.error("agent %s not in swarm (have %s)", agent_id, list(self.swarm))
            raise

    def backend_for(self, agent_id: str) -> InferenceBackend:
        cfg = self.agent(agent_id)
        target = cfg.engine or cfg.server_group or ""
        try:
            return self.backends[target]
        except KeyError as exc:
            logger.error("no backend for %s (target=%r, have %s)",
                         agent_id, target, list(self.backends))
            raise


class OrchestrationMode(ABC):
    """Subclass and set `mode_id` (matches filename slug)."""

    mode_id: str = ""

    @abstractmethod
    async def execute(self, ctx: ModeContext, query: str) -> AsyncIterator[Event]:
        """Yield Events. Implementations must yield at least one final `result` event."""
