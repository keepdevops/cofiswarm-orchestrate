"""POST /api/orchestrate — Python-backend orchestration mode dispatcher.

Registered in service.make_app() alongside /api/mlx/* routes.
Accepts {mode, prompt, params, session_id} and dispatches to the
appropriate Python OrchestrationMode, returning blocking JSON.
SSE streaming passthrough is added in MS-25-2.

Modes registered here: map_reduce, speculative, critic_debate.
tree_of_thought is stretch and registered once MS-25-4 caps prove stable.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import aiohttp
from aiohttp import web

from cofiswarm_orchestrate.modes.base import ModeContext
from cofiswarm_orchestrate.modes.critic_debate import CriticDebateMode
from cofiswarm_orchestrate.modes.map_reduce import MapReduceMode
from cofiswarm_orchestrate.modes.speculative import SpeculativeMode
from cofiswarm_orchestrate.modes.tree_of_thought import TreeOfThoughtMode
from cofiswarm_orchestrate.memory_utils import check_mode_memory_ok

logger = logging.getLogger(__name__)

_RAG_BASE = (
    os.environ.get("RAG_INGEST_HOST", "http://127.0.0.1")
    + ":"
    + os.environ.get("RAG_INGEST_PORT", "8001")
)


async def _fetch_rag_chunks(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Call the RAG ingest service /retrieve and return chunk dicts."""
    url = f"{_RAG_BASE}/retrieve"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json={"query": query, "k": k}, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    logger.error("rag retrieve HTTP %d from %s", r.status, url)
                    return []
                data = await r.json()
                return data.get("chunks") or []
    except Exception as exc:
        logger.error("rag retrieve failed (non-fatal): %s", exc)
        return []


async def _rag_context_for(body: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """Retrieve top-k RAG chunks for a request, filtered by min_score.

    ``min_score`` is the maximum cosine distance to accept (1.0 = no filter),
    matching the C++ coordinator (rag_client.cpp) and the UI tooltip. Returns []
    when RAG is disabled or nothing passes the filter.
    """
    if not body.get("use_rag"):
        return []
    k = int(body.get("rag_top_k") or 3)
    min_score = float(body.get("rag_min_score", 1.0))
    chunks = await _fetch_rag_chunks(prompt, k=k)
    return [c for c in chunks if float(c.get("distance", 0.0)) <= min_score]


def _roster_rag_default(swarm: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Bridge the roster's per-agent use_rag into the MLX path. When the request is
    SILENT on use_rag, enable it if any swarm agent opts in (use_rag), defaulting
    rag_top_k to the max across opted-in agents. An explicit request use_rag (true or
    false) always wins, so the UI toggle is never overridden when it sends one."""
    if "use_rag" in body:
        return body
    opted = [a for a in swarm.values() if getattr(a, "use_rag", False)]
    if not opted:
        return body
    out = {**body, "use_rag": True}
    if not out.get("rag_top_k"):
        topk = max((getattr(a, "rag_top_k", None) or 0) for a in opted)
        if topk:
            out["rag_top_k"] = topk
    return out


_PYTHON_MODES = {m.mode_id: m for m in [MapReduceMode(), SpeculativeMode(), CriticDebateMode(), TreeOfThoughtMode()]}


async def handle_orchestrate(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError(f"expected JSON object, got {type(body).__name__}")
    except Exception as exc:
        logger.error("orchestrate: bad JSON: %s", exc)
        raise web.HTTPBadRequest(reason="invalid JSON")

    mode_id = (body.get("mode") or "").strip()
    if not mode_id:
        raise web.HTTPBadRequest(reason="'mode' required")

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise web.HTTPBadRequest(reason="'prompt' required")

    mode = _PYTHON_MODES.get(mode_id)
    if mode is None:
        raise web.HTTPBadRequest(
            reason=f"unknown Python mode {mode_id!r} — choose from {list(_PYTHON_MODES)}"
        )

    mem_ok, mem_err = check_mode_memory_ok(mode_id)
    if not mem_ok:
        logger.warning("orchestrate: memory guard blocked mode=%s: %s", mode_id, mem_err)
        raise web.HTTPServiceUnavailable(reason=mem_err)

    session_id = (body.get("session_id") or "").strip() or str(uuid.uuid4())
    params: dict[str, Any] = body.get("params") or {}

    # Bridge per-agent use_rag from the roster when the request is silent (UI omits it
    # when its RAG toggle is off; an explicit request value still wins).
    body = _roster_rag_default(request.app["swarm"], body)
    # Retrieve top-k chunks (filtered by min_score) into params["rag_context"]; the modes
    # inject it per-agent via rag_xml() in their prompt templates.
    rag_chunks = await _rag_context_for(body, prompt)
    if rag_chunks:
        params = {**params, "rag_context": rag_chunks}

    try:
        ctx = ModeContext(
            swarm=request.app["swarm"],
            backends=request.app["backends"],
            agents=list(request.app["swarm"].keys()),
            params=params,
            request_id=session_id,
        )
        parts: list[str] = []
        meta: dict[str, Any] = {}
        async for event in mode.execute(ctx, prompt):
            if event.kind == "token":
                parts.append(event.text)
            elif event.kind == "result" and event.meta:
                meta = dict(event.meta)
            elif event.kind == "error":
                logger.error("orchestrate: mode=%s agent=%s error: %s",
                             mode_id, event.agent_id, event.text)
    except Exception as exc:
        logger.error("orchestrate: mode=%s session=%s failed: %s", mode_id, session_id, exc)
        raise web.HTTPInternalServerError(reason=str(exc))

    rag_context = params.get("rag_context") or []
    if rag_context:
        meta = {**meta, "rag_chunks": rag_context}
    return web.json_response(
        {"result": "".join(parts), "session_id": session_id, "mode": mode_id, "meta": meta}
    )


async def handle_orchestrate_stream(request: web.Request) -> web.StreamResponse:
    """POST /api/orchestrate/stream — SSE streaming variant of handle_orchestrate."""
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError(f"expected JSON object, got {type(body).__name__}")
    except Exception as exc:
        logger.error("orchestrate/stream: bad JSON: %s", exc)
        raise web.HTTPBadRequest(reason="invalid JSON")

    mode_id = (body.get("mode") or "").strip()
    prompt = (body.get("prompt") or "").strip()
    if not mode_id:
        raise web.HTTPBadRequest(reason="'mode' required")
    if not prompt:
        raise web.HTTPBadRequest(reason="'prompt' required")

    mode = _PYTHON_MODES.get(mode_id)
    if mode is None:
        raise web.HTTPBadRequest(
            reason=f"unknown Python mode {mode_id!r} — choose from {list(_PYTHON_MODES)}"
        )

    mem_ok, mem_err = check_mode_memory_ok(mode_id)
    if not mem_ok:
        logger.warning("orchestrate/stream: memory guard blocked mode=%s: %s", mode_id, mem_err)
        raise web.HTTPServiceUnavailable(reason=mem_err)

    session_id = (body.get("session_id") or "").strip() or str(uuid.uuid4())
    params: dict[str, Any] = body.get("params") or {}

    body = _roster_rag_default(request.app["swarm"], body)  # roster->MLX use_rag bridge
    rag_chunks = await _rag_context_for(body, prompt)
    if rag_chunks:
        params = {**params, "rag_context": rag_chunks}  # modes inject it per-agent via rag_xml()

    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
    })
    await resp.prepare(request)

    async def send(event: str, data: str) -> None:
        await resp.write(f"event: {event}\ndata: {data}\n\n".encode())

    result_parts: list[str] = []
    final_meta: dict[str, Any] = {}
    token_counts: dict[str, int] = {}   # agent_id → word-approximate token count
    agent_elapsed: dict[str, float] = {}  # agent_id → accumulated ms
    agent_start_ts: dict[str, float] = {}  # agent_id → monotonic start time
    try:
        ctx = ModeContext(
            swarm=request.app["swarm"],
            backends=request.app["backends"],
            agents=list(request.app["swarm"].keys()),
            params=params,
            request_id=session_id,
        )
        async for event in mode.execute(ctx, prompt):
            if event.kind == "token":
                result_parts.append(event.text)
                if event.agent_id:
                    token_counts[event.agent_id] = (
                        token_counts.get(event.agent_id, 0) + len(event.text.split())
                    )
                await send("token", json.dumps(
                    {"agent_id": event.agent_id, "text": event.text}))
            elif event.kind == "agent_start":
                if event.agent_id:
                    agent_start_ts[event.agent_id] = time.monotonic()
                await send("agent_start", json.dumps(
                    {"agent_id": event.agent_id, "meta": event.meta}))
            elif event.kind == "agent_end":
                if event.agent_id and event.agent_id in agent_start_ts:
                    elapsed_ms = (time.monotonic() - agent_start_ts.pop(event.agent_id)) * 1000
                    agent_elapsed[event.agent_id] = (
                        agent_elapsed.get(event.agent_id, 0.0) + elapsed_ms
                    )
                await send("agent_end", json.dumps(
                    {"agent_id": event.agent_id}))
            elif event.kind == "result" and event.meta:
                final_meta = dict(event.meta)
            elif event.kind == "error":
                logger.error("orchestrate/stream: mode=%s agent=%s error: %s",
                             mode_id, event.agent_id, event.text)
                await send("error", json.dumps(
                    {"agent_id": event.agent_id, "error": event.text}))
    except Exception as exc:
        logger.error("orchestrate/stream: mode=%s session=%s failed: %s",
                     mode_id, session_id, exc)
        await send("error", json.dumps({"agent_id": None, "error": str(exc)}))
        return resp

    rag_context = params.get("rag_context") or []
    if rag_context:
        final_meta = {**final_meta, "rag_chunks": rag_context}
    all_agent_ids = set(token_counts) | set(agent_elapsed)
    if all_agent_ids:
        final_meta = {
            **final_meta,
            "timings": {
                agent_id: {
                    "completion_tokens": token_counts.get(agent_id, 0),
                    "total_ms": int(agent_elapsed.get(agent_id, 0.0)),
                }
                for agent_id in all_agent_ids
            },
        }
    await send("done", json.dumps({
        "result": "".join(result_parts),
        "session_id": session_id,
        "mode": mode_id,
        "meta": final_meta,
    }))
    return resp


def register_orchestrate_routes(app: web.Application) -> None:
    app.router.add_post("/api/orchestrate", handle_orchestrate)
    app.router.add_post("/api/orchestrate/stream", handle_orchestrate_stream)
