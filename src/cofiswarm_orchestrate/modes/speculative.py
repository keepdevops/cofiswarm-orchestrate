"""Speculative consensus — small drafter generates, large verifier confirms.

The drafter (typically a fast local MLX model) emits a block of tokens; the
verifier (typically a remote vLLM model) re-scores them in a single pass.
This is a coarse-grained Python implementation: when a token diverges, we
accept the verifier's correction and continue.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from backends.base import GenerateRequest
from .base import Event, ModeContext, OrchestrationMode
from ._helpers import rag_xml

logger = logging.getLogger(__name__)


class SpeculativeMode(OrchestrationMode):
    mode_id = "speculative"

    async def execute(self, ctx: ModeContext, query: str) -> AsyncIterator[Event]:
        drafter_id = ctx.params.get("drafter")
        verifier_id = ctx.params.get("verifier")
        if not drafter_id or not verifier_id:
            logger.error("speculative: missing drafter/verifier in params (got %s)",
                         ctx.params)
            raise ValueError("speculative: drafter and verifier required in params")

        drafter_cfg = ctx.agent(drafter_id)
        verifier_cfg = ctx.agent(verifier_id)
        drafter = ctx.backend_for(drafter_id)
        verifier = ctx.backend_for(verifier_id)

        block_size = int(ctx.params.get("block_size", 32))
        rag_block = rag_xml(ctx.params.get("rag_context") or [])
        prompt = f"<system>{verifier_cfg.system_prompt}</system>\n{rag_block}<query>{query}</query>"
        emitted: list[str] = []
        max_total = verifier_cfg.max_tokens

        yield Event(kind="agent_start", agent_id=drafter_id, meta={"role": "drafter"})
        yield Event(kind="agent_start", agent_id=verifier_id, meta={"role": "verifier"})

        while len("".join(emitted)) < max_total:
            draft: list[str] = []
            try:
                async for chunk in drafter.generate_stream(
                    GenerateRequest(
                        prompt=prompt + "".join(emitted),
                        max_tokens=block_size,
                    )
                ):
                    if chunk.text:
                        draft.append(chunk.text)
                    if sum(len(d) for d in draft) >= block_size:
                        break
            except Exception as exc:
                logger.error("speculative drafter failed: %s", exc)
                yield Event(kind="error", agent_id=drafter_id, text=str(exc))
                return

            if not draft:
                break

            draft_text = "".join(draft)
            # Verifier re-generates from the same prefix; on divergence, accept verifier.
            verified: list[str] = []
            try:
                async for chunk in verifier.generate_stream(
                    GenerateRequest(
                        prompt=prompt + "".join(emitted),
                        max_tokens=len(draft_text),
                    )
                ):
                    if chunk.text:
                        verified.append(chunk.text)
                    if sum(len(v) for v in verified) >= len(draft_text):
                        break
            except Exception as exc:
                logger.error("speculative verifier failed: %s", exc)
                yield Event(kind="error", agent_id=verifier_id, text=str(exc))
                return

            v_text = "".join(verified) or draft_text
            emitted.append(v_text)
            yield Event(kind="token", agent_id=verifier_id, text=v_text,
                        meta={"draft_match": v_text == draft_text})
            if len(v_text) < block_size:
                break  # short response — model is done

        yield Event(kind="agent_end", agent_id=drafter_id)
        yield Event(kind="agent_end", agent_id=verifier_id)
        yield Event(
            kind="result",
            text="".join(emitted),
            meta={"mode": "speculative", "drafter": drafter_id, "verifier": verifier_id},
        )
