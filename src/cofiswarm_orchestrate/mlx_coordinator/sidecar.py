"""MS-142: Orchestrate-only sidecar — thin aiohttp process for /api/orchestrate*.

Runs on ORCH_SIDECAR_PORT (default 3003). Handles map_reduce, speculative,
critic_debate, tree_of_thought modes. No mlx dependency; all agents speak
the OpenAI /v1/chat/completions HTTP API so MlxBackend works for any engine.

Started by brewctl launch; stopped by brewctl shutdown.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from aiohttp import web

from cofiswarm_orchestrate import observer
from cofiswarm_orchestrate.manager import SwarmFactory
from cofiswarm_orchestrate.mlx_coordinator.backend import MlxBackend
from cofiswarm_orchestrate.mlx_coordinator.service_orchestrate import register_orchestrate_routes

logger = logging.getLogger(__name__)

_DEFAULT_PORT = int(os.environ.get("ORCH_SIDECAR_PORT", "3003"))


# ---------------------------------------------------------------------------
# CORS middleware (same policy as service.py)
# ---------------------------------------------------------------------------

def _cors(resp: web.StreamResponse) -> web.StreamResponse:
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@web.middleware
async def cors_mw(request: web.Request, handler: Any) -> web.StreamResponse:
    if request.method == "OPTIONS":
        return _cors(web.Response(status=204))
    try:
        resp = await handler(request)
    except web.HTTPException as exc:
        _cors(exc)
        raise
    return _cors(resp)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _build_backends(swarm: dict) -> dict:
    """One MlxBackend per agent — works for any engine (llama/mlx speak same API)."""
    backends: dict[str, MlxBackend] = {}
    for agent_id, cfg in swarm.items():
        key = cfg.engine or cfg.server_group or agent_id
        backends[key] = MlxBackend(
            port=cfg.port or 8083,
            agent_id=agent_id,
            system_prompt=cfg.system_prompt,
            max_tokens=cfg.max_tokens,
        )
    return backends


def make_sidecar_app() -> web.Application:
    app = web.Application(middlewares=[cors_mw])

    async def on_startup(a: web.Application) -> None:
        factory = SwarmFactory()
        swarm = factory.load_swarm()
        if not swarm:
            logger.error("orch-sidecar: no agents found in config/agents/")
            raise RuntimeError("no agents configured")
        a["swarm"] = swarm
        a["backends"] = _build_backends(swarm)
        logger.info("orch-sidecar: loaded %d agents for orchestrate modes", len(swarm))

    async def on_cleanup(a: web.Application) -> None:
        for backend in a["backends"].values():
            await backend.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    register_orchestrate_routes(app)
    observer.attach(app)  # announce presence on the observer bus (no-op unless COFISWARM_NATS_URL set)
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=os.environ.get("ORCH_SIDECAR_LOG", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    p = argparse.ArgumentParser(prog="orch-sidecar")
    p.add_argument("--host", default=os.environ.get("ORCH_SIDECAR_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = p.parse_args()
    logger.info("orch-sidecar: starting on %s:%d", args.host, args.port)
    web.run_app(make_sidecar_app(), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
