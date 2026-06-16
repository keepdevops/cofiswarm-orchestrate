"""Critic/debate — a Generator proposes, a Critic reviews, repeat until SHIP or limit."""
from __future__ import annotations

import logging
from typing import AsyncIterator

from backends.base import GenerateRequest
from .base import Event, ModeContext, OrchestrationMode
from ._helpers import rag_xml

logger = logging.getLogger(__name__)


class CriticDebateMode(OrchestrationMode):
    mode_id = "critic_debate"

    async def execute(self, ctx: ModeContext, query: str) -> AsyncIterator[Event]:
        gen_id = ctx.params.get("generator")
        critic_id = ctx.params.get("critic")
        if not gen_id or not critic_id:
            logger.error("critic_debate: generator and critic required (params=%s)",
                         ctx.params)
            raise ValueError("critic_debate: generator and critic required")
        max_rounds = int(ctx.params.get("max_rounds", 3))

        gen_cfg = ctx.agent(gen_id)
        critic_cfg = ctx.agent(critic_id)
        gen_backend = ctx.backend_for(gen_id)
        critic_backend = ctx.backend_for(critic_id)

        rag_block = rag_xml(ctx.params.get("rag_context") or [])
        current: str = ""
        critique: str = ""
        for round_idx in range(1, max_rounds + 1):
            gen_prompt = (
                f"<system>{gen_cfg.system_prompt}</system>\n{rag_block}"
                f"<query>{query}</query>\n"
                f"<prior_attempt>{current}</prior_attempt>\n"
                f"<critique>{critique}</critique>"
            )
            yield Event(kind="agent_start", agent_id=gen_id,
                        meta={"role": "generator", "round": round_idx})
            buf: list[str] = []
            try:
                async for c in gen_backend.generate_stream(
                    GenerateRequest(prompt=gen_prompt, max_tokens=gen_cfg.max_tokens)
                ):
                    if c.text:
                        buf.append(c.text)
                        yield Event(kind="token", agent_id=gen_id, text=c.text)
            except Exception as exc:
                logger.error("critic_debate: generator %s failed: %s", gen_id, exc)
                yield Event(kind="error", agent_id=gen_id, text=str(exc))
                return
            current = "".join(buf)
            yield Event(kind="agent_end", agent_id=gen_id)

            critic_prompt = (
                f"<system>{critic_cfg.system_prompt}</system>\n{rag_block}"
                f"<query>{query}</query>\n<proposal>{current}</proposal>\n"
                f"<task>Reply with SHIP if acceptable, otherwise REWRITE followed by "
                f"specific critiques.</task>"
            )
            yield Event(kind="agent_start", agent_id=critic_id,
                        meta={"role": "critic", "round": round_idx})
            cbuf: list[str] = []
            try:
                async for c in critic_backend.generate_stream(
                    GenerateRequest(prompt=critic_prompt, max_tokens=critic_cfg.max_tokens)
                ):
                    if c.text:
                        cbuf.append(c.text)
                        yield Event(kind="token", agent_id=critic_id, text=c.text)
            except Exception as exc:
                logger.error("critic_debate: critic %s failed: %s", critic_id, exc)
                yield Event(kind="error", agent_id=critic_id, text=str(exc))
                return
            critique = "".join(cbuf)
            yield Event(kind="agent_end", agent_id=critic_id)

            if critique.strip().upper().startswith("SHIP"):
                yield Event(
                    kind="result",
                    text=current,
                    meta={"mode": "critic_debate", "rounds": round_idx, "verdict": "SHIP"},
                )
                return

        yield Event(
            kind="result",
            text=current,
            meta={"mode": "critic_debate", "rounds": max_rounds, "verdict": "MAX_ROUNDS"},
        )
