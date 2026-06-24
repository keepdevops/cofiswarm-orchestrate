"""SwarmFactory — loads per-agent JSON configs from config/agents/ at startup.

Validates each config against AgentConfig (Pydantic) and fails loudly on missing
required fields. See CLAUDE.md §2 (Fail Loudly).
"""
from __future__ import annotations

import glob
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any


def _default_model_dir() -> str:
    """Fallback for MATRIX_MODEL_DIR, mirroring scripts/matrix-env.sh."""
    if platform.system() == "Darwin":
        return "/Users/Shared/llama/models"
    return ""


os.environ.setdefault("MATRIX_MODEL_DIR", _default_model_dir())

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(os.environ.get("COFISWARM_CONFIG_ROOT", "/etc/cofiswarm/config")).parent
DEFAULT_AGENTS_DIR = REPO_ROOT / "config" / "agents"


class AgentConfig(BaseModel):
    """Schema for a single agent file under config/agents/<slug>.json."""

    agent_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    context: int = Field(gt=0)
    max_tokens: int = Field(gt=0)

    # Optional engine/transport fields preserved from legacy swarm-config.json.
    engine: str | None = None
    server_group: str | None = None
    port: int | None = None
    gpu_layers: int | None = None
    n_batch: int | None = None
    read_timeout_secs: int | None = None
    max_concurrency: int | None = None

    # Routing barrier: "mlx" means this agent is owned by the Python MLX coordinator,
    # not the C++ coordinator. The C++ side skips these agents at configure time.
    coordinator: str | None = None

    # Per-agent RAG targeting (mirrors swarm-config.json / agent-registry). When set,
    # the MLX orchestrate path defaults use_rag on for runs over this agent.
    use_rag: bool = False
    rag_top_k: int | None = None

    # Future fields (Phase 4): RAG config.
    rag: dict[str, Any] | None = None

    @field_validator("agent_id")
    @classmethod
    def _slug_only(cls, v: str) -> str:
        if any(c.isspace() for c in v):
            raise ValueError("agent_id must be slug-form (no whitespace)")
        return v

    @field_validator("model")
    @classmethod
    def _expand_model_path(cls, v: str) -> str:
        expanded = os.path.expandvars(os.path.expanduser(v))
        if "$" in expanded:
            # An unresolved ${VAR} would silently break llama-server at launch.
            logger.error("unresolved env var in model path: %s", v)
            raise ValueError(f"unresolved env var in model path: {v}")
        return expanded


class SwarmFactory:
    """Discovers and loads agent configurations into an in-memory registry."""

    def __init__(self, agents_dir: os.PathLike[str] | str = DEFAULT_AGENTS_DIR) -> None:
        self.agents_dir = Path(agents_dir)
        self.active_swarm: dict[str, AgentConfig] = {}

    def load_swarm(self) -> dict[str, AgentConfig]:
        if not self.agents_dir.is_dir():
            logger.error("agents dir missing: %s", self.agents_dir)
            raise FileNotFoundError(self.agents_dir)

        pattern = str(self.agents_dir / "*.json")
        paths = sorted(glob.glob(pattern))
        if not paths:
            logger.error("no agent JSON files under %s", self.agents_dir)
            raise RuntimeError(f"no agents found in {self.agents_dir}")

        for path in paths:
            try:
                raw = json.loads(Path(path).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("failed to read agent config %s: %s", path, exc)
                raise

            try:
                cfg = AgentConfig.model_validate(raw)
            except ValidationError as exc:
                logger.error("invalid agent config %s: %s", path, exc)
                raise

            if cfg.agent_id in self.active_swarm:
                logger.error("duplicate agent_id %s in %s", cfg.agent_id, path)
                raise RuntimeError(f"duplicate agent_id: {cfg.agent_id}")

            self.active_swarm[cfg.agent_id] = cfg

        logger.info("[BOOT] loaded %d isolated agents from %s",
                    len(self.active_swarm), self.agents_dir)
        return self.active_swarm
