"""Tree-of-thought — generate K candidate branches, score them, recurse on the best.

A scorer agent rates each branch on a 0-10 scale; branches below `prune_below`
are dropped. We descend `depth` levels, then return the highest-scored leaf.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator

from backends.base import GenerateRequest
from .base import Event, ModeContext, OrchestrationMode
from ._helpers import rag_xml

logger = logging.getLogger(__name__)


class TreeOfThoughtMode(OrchestrationMode):
    mode_id = "tree_of_thought"

    async def execute(self, ctx: ModeContext, query: str) -> AsyncIterator[Event]:
        gen_id = ctx.params.get("generator") or (ctx.agents[0] if ctx.agents else None)
        scorer_id = ctx.params.get("scorer") or (
            ctx.agents[1] if len(ctx.agents) > 1 else gen_id
        )
        if not gen_id or not scorer_id:
            logger.error("tree_of_thought: need generator+scorer (params=%s agents=%s)",
                         ctx.params, ctx.agents)
            raise ValueError("tree_of_thought: generator and scorer required")
        depth = int(ctx.params.get("depth", 2))
        branching = int(ctx.params.get("branching", 3))
        prune_below = float(ctx.params.get("prune_below", 4.0))

        gen_cfg = ctx.agent(gen_id)
        scorer_cfg = ctx.agent(scorer_id)
        gen_backend = ctx.backend_for(gen_id)
        scorer_backend = ctx.backend_for(scorer_id)
        rag_block = rag_xml(ctx.params.get("rag_context") or [])

        async def gen_branch(prefix: str, seed: int) -> str:
            prompt = (
                f"<system>{gen_cfg.system_prompt}</system>\n{rag_block}"
                f"<query>{query}</query>\n"
                f"<prior>{prefix}</prior>\n<seed>{seed}</seed>\n"
                f"<task>Continue with one alternative next step.</task>"
            )
            buf: list[str] = []
            try:
                async for c in gen_backend.generate_stream(
                    GenerateRequest(prompt=prompt, max_tokens=gen_cfg.max_tokens // 2)
                ):
                    if c.text:
                        buf.append(c.text)
            except Exception as exc:
                logger.error("tot: gen failed: %s", exc)
                return ""
            return "".join(buf)

        async def score(text: str) -> float:
            prompt = (
                f"<system>{scorer_cfg.system_prompt}</system>\n{rag_block}"
                f"<query>{query}</query>\n"
                f"<candidate>{text}</candidate>\n<task>Reply with a single number 0-10.</task>"
            )
            buf: list[str] = []
            try:
                async for c in scorer_backend.generate_stream(
                    GenerateRequest(prompt=prompt, max_tokens=8)
                ):
                    if c.text:
                        buf.append(c.text)
            except Exception as exc:
                logger.error("tot: score failed: %s", exc)
                return 0.0
            m = re.search(r"(\d+(?:\.\d+)?)", "".join(buf))
            return float(m.group(1)) if m else 0.0

        best_prefix = ""
        for d in range(depth):
            yield Event(kind="agent_start", agent_id=gen_id,
                        meta={"role": "generator", "depth": d})
            branches = await asyncio.gather(
                *[gen_branch(best_prefix, i) for i in range(branching)]
            )
            yield Event(kind="agent_end", agent_id=gen_id)

            yield Event(kind="agent_start", agent_id=scorer_id,
                        meta={"role": "scorer", "depth": d})
            scores = await asyncio.gather(*[score(best_prefix + b) for b in branches])
            yield Event(kind="agent_end", agent_id=scorer_id)

            ranked = sorted(zip(scores, branches), key=lambda p: p[0], reverse=True)
            kept = [(s, b) for s, b in ranked if s >= prune_below] or ranked[:1]
            top_score, top_branch = kept[0]
            yield Event(
                kind="token",
                agent_id=gen_id,
                text=top_branch,
                meta={"depth": d, "score": top_score, "pruned": len(branches) - len(kept)},
            )
            best_prefix = best_prefix + top_branch
            if not top_branch:
                break

        yield Event(
            kind="result",
            text=best_prefix,
            meta={"mode": "tree_of_thought", "depth": depth, "branching": branching},
        )
