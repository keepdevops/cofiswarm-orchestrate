#!/usr/bin/env python3
"""Regression guard: RAG context is injected EXACTLY ONCE on the MLX orchestrate path.

The orchestration modes inject ``params["rag_context"]`` per-agent via ``rag_xml()``
in their prompt templates. A prior bug ALSO prepended a ``<context source="rag">``
block to the prompt in service_orchestrate, double-injecting the context. This guard
fails if either (a) a mode stops injecting / injects twice, or (b) service_orchestrate
re-introduces its own block prepend.

Standalone (no pytest), mirroring cofiswarm-rag's store_sqlite_check.py convention.
Asserts loudly (CLAUDE.md §2); exits non-zero on any failure.
"""
from __future__ import annotations

import asyncio
import inspect
import sys

from backends.base import TokenChunk
from cofiswarm_orchestrate.manager import AgentConfig
from cofiswarm_orchestrate.modes.base import ModeContext
from cofiswarm_orchestrate.modes.map_reduce import MapReduceMode
import cofiswarm_orchestrate.mlx_coordinator.service_orchestrate as svc


class _FakeBackend:
    """Captures the prompt each agent call receives."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate_stream(self, req):  # noqa: ANN001
        self.prompts.append(req.prompt)
        yield TokenChunk(text="ok", done=True)

    async def close(self) -> None:
        pass


def _agent(slug: str) -> AgentConfig:
    return AgentConfig(
        agent_id=slug, name=slug, model="/x/model", system_prompt=f"You are {slug}.",
        context=2048, max_tokens=32, engine="mlx",
    )


async def _mode_injects_once() -> None:
    chunk = {"content": "DOCBODY kvrouter eviction", "source_path": "u://d",
             "chunk_idx": 0, "distance": 0.55}
    backend = _FakeBackend()
    ctx = ModeContext(
        swarm={"scout": _agent("scout"), "synth": _agent("synth")},
        backends={"mlx": backend},
        agents=["scout", "synth"],
        params={"rag_context": [chunk], "chunks": ["review the eviction logic"]},
        request_id="t",
    )
    async for _ in MapReduceMode().execute(ctx, "kvrouter slot eviction"):
        pass

    assert backend.prompts, "no agent prompts captured — mode did not call any backend"
    for p in backend.prompts:
        n_block = p.count("<retrieved>")
        n_body = p.count("DOCBODY kvrouter eviction")
        assert n_block == 1, f"expected one <retrieved> block, got {n_block}:\n{p}"
        assert n_body == 1, f"chunk content injected {n_body}x (double-inject?):\n{p}"
        assert '<context source="rag">' not in p, \
            f"stray <context source=rag> block (the reverted double-inject):\n{p}"
    print(f"  ok: map_reduce injected rag_context exactly once across {len(backend.prompts)} agent call(s)")


def _service_has_no_own_block() -> None:
    assert not hasattr(svc, "_render_rag_block"), \
        "service_orchestrate must not render its own RAG block — modes inject via rag_xml()"
    source = inspect.getsource(svc)
    assert '<context source="rag">' not in source, \
        "double-inject regression: service_orchestrate re-introduced a RAG block prepend"
    print("  ok: service_orchestrate has no own RAG-block render/prepend")


def _bridge_semantics() -> None:
    class A:  # minimal AgentConfig stand-in
        def __init__(self, u, k):
            self.use_rag, self.rag_top_k = u, k

    swarm = {"a": A(True, 5), "b": A(False, None)}
    assert svc._roster_rag_default(swarm, {"mode": "flat"}) == {"mode": "flat", "use_rag": True, "rag_top_k": 5}, \
        "silent request should default use_rag+top_k from the roster"
    assert svc._roster_rag_default(swarm, {"use_rag": False}) == {"use_rag": False}, \
        "explicit use_rag=false must win"
    assert svc._roster_rag_default({"x": A(False, None)}, {"mode": "flat"}) == {"mode": "flat"}, \
        "no opted-in agents should leave the request unchanged"
    print("  ok: _roster_rag_default bridge semantics (silent->roster, explicit-wins, no-opt->unchanged)")


def main() -> int:
    asyncio.run(_mode_injects_once())
    _service_has_no_own_block()
    _bridge_semantics()
    print("ok: MLX RAG inject-once regression guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
