"""Attach orchestrate to the NATS observer bus via the shared cofiswarm-observer-sdk Python
ServiceComponent: announce presence and serve .orchestrate.{info,health}.

Default-off: enabled only when COFISWARM_NATS_URL is set, mirroring the opt-in bus mode of the
other components. The SDK is imported lazily, so the sidecar runs identically — and without
cofiswarm-observer-sdk installed — whenever the bus is disabled.
"""
from __future__ import annotations

import logging
import os

from aiohttp import web

logger = logging.getLogger(__name__)


def attach(app: web.Application) -> None:
    """Wire observer presence into the aiohttp app lifecycle. No-op unless COFISWARM_NATS_URL
    is set; never imports the SDK or touches the bus when disabled."""
    url = os.environ.get("COFISWARM_NATS_URL")
    if not url:
        logger.info("observer: COFISWARM_NATS_URL unset; bus attach disabled")
        return

    try:
        from cofiswarm_observer import ServiceComponent, contract
    except ImportError as exc:  # loud: asked for the bus but the client isn't installed
        logger.error("observer: COFISWARM_NATS_URL set but cofiswarm-observer-sdk missing: %s", exc)
        return

    subj_info = f"{contract.PREFIX}.orchestrate.info"
    subj_health = f"{contract.PREFIX}.orchestrate.health"

    async def info(_req: dict) -> dict:
        return {
            "component": "orchestrate",
            "modes": ["map_reduce", "speculative", "critic_debate", "tree_of_thought"],
        }

    async def health(_req: dict) -> dict:
        return {"status": "ok"}

    routes = {subj_info: info, subj_health: health}

    async def on_startup(a: web.Application) -> None:
        try:
            nc = await ServiceComponent.connect(url, "cofiswarm-orchestrate")
        except Exception as exc:  # loud: never silently run detached from the bus
            logger.error("observer: NATS connect %s failed: %s", url, exc)
            return
        comp = ServiceComponent(nc, "orchestrate", "orchestrate", routes)
        await comp.start()
        a["observer_nc"] = nc
        a["observer_comp"] = comp
        logger.info("observer: orchestrate announced on %s (.orchestrate.info/.health)", url)

    async def on_cleanup(a: web.Application) -> None:
        comp = a.get("observer_comp")
        if comp is not None:
            await comp.shutdown()  # goodbye -> offline
        nc = a.get("observer_nc")
        if nc is not None:
            await nc.close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
