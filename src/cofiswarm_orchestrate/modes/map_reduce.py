"""Map-reduce — split input chunks across agents, then synthesize the findings.

`chunks` in ctx.params is a list[str]; each chunk is mapped through one of the
configured worker agents (round-robin), then a synthesizer agent merges the
per-chunk outputs into a single response.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from backends.base import GenerateRequest
from .base import Event, ModeContext, OrchestrationMode
from ._helpers import rag_xml

logger = logging.getLogger(__name__)


class MapReduceMode(OrchestrationMode):
    mode_id = "map_reduce"

    async def execute(self, ctx: ModeContext, query: str) -> AsyncIterator[Event]:
        chunks: list[str] = list(ctx.params.get("chunks") or [])
        if not chunks:
            logger.error("map_reduce: empty chunks list (request_id=%s)", ctx.request_id)
            raise ValueError("map_reduce: 'chunks' param required")
        if not ctx.agents:
            raise ValueError("map_reduce: empty agents list")
        synthesizer = ctx.params.get("synthesizer") or ctx.agents[-1]
        workers = [a for a in ctx.agents if a != synthesizer] or ctx.agents
        rag_block = rag_xml(ctx.params.get("rag_context") or [])

        async def map_one(idx: int, chunk: str) -> tuple[int, str, str | None]:
            agent_id = workers[idx % len(workers)]
            cfg = ctx.agent(agent_id)
            backend = ctx.backend_for(agent_id)
            prompt = (
                f"<system>{cfg.system_prompt}</system>\n"
                f"{rag_block}"
                f"<query>{query}</query>\n<chunk idx={idx}>\n{chunk}\n</chunk>"
            )
            buf: list[str] = []
            try:
                async for c in backend.generate_stream(
                    GenerateRequest(prompt=prompt, max_tokens=cfg.max_tokens)
                ):
                    if c.text:
                        buf.append(c.text)
                return idx, "".join(buf), None
            except Exception as exc:
                logger.error("map_reduce worker %s on chunk %d failed: %s",
                             agent_id, idx, exc)
                return idx, "", str(exc)

        for idx in range(len(chunks)):
            yield Event(kind="agent_start", agent_id=workers[idx % len(workers)],
                        meta={"phase": "map", "chunk": idx})

        results = await asyncio.gather(*[map_one(i, c) for i, c in enumerate(chunks)])

        mapped: list[str] = []
        for idx, text, err in results:
            if err:
                yield Event(kind="error", agent_id=workers[idx % len(workers)], text=err)
            else:
                yield Event(kind="token", agent_id=workers[idx % len(workers)],
                            text=text, meta={"phase": "map", "chunk": idx})
            mapped.append(text)

        synth_cfg = ctx.agent(synthesizer)
        synth_backend = ctx.backend_for(synthesizer)
        joined = "\n\n".join(
            f"<finding chunk={i}>\n{t}\n</finding>" for i, t in enumerate(mapped) if t
        )
        synth_prompt = (
            f"<system>{synth_cfg.system_prompt}</system>\n"
            f"{rag_block}"
            f"<query>{query}</query>\n"
            f"<findings>\n{joined}\n</findings>"
        )

        yield Event(kind="agent_start", agent_id=synthesizer,
                    meta={"phase": "reduce", "role": "synthesizer"})
        final: list[str] = []
        try:
            async for c in synth_backend.generate_stream(
                GenerateRequest(prompt=synth_prompt, max_tokens=synth_cfg.max_tokens)
            ):
                if c.text:
                    final.append(c.text)
                    yield Event(kind="token", agent_id=synthesizer, text=c.text)
        except Exception as exc:
            logger.error("map_reduce synth %s failed: %s", synthesizer, exc)
            yield Event(kind="error", agent_id=synthesizer, text=str(exc))
            return
        yield Event(kind="agent_end", agent_id=synthesizer)

        yield Event(
            kind="result",
            text="".join(final),
            meta={"mode": "map_reduce", "n_chunks": len(chunks),
                  "synthesizer": synthesizer},
        )
