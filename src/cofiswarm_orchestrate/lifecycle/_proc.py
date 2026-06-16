"""Process helpers — thin wrappers around lsof/pgrep/kill for the lifecycle CLI."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time

logger = logging.getLogger(__name__)


def lsof_pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:
        logger.error("lsof not available: %s", exc)
        return []
    return [int(p) for p in out.stdout.split() if p.strip().isdigit()]


def pgrep_pids(pattern: str) -> list[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as exc:
        logger.error("pgrep not available: %s", exc)
        return []
    return [int(p) for p in out.stdout.split() if p.strip().isdigit()]


def _send(pids: list[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            logger.error("kill %d (%s): %s", pid, sig, exc)


def kill_pids(pids: list[int], *, term_wait: float = 3.0) -> list[int]:
    """SIGTERM, wait up to term_wait seconds, then SIGKILL any survivors.
    Returns the list of PIDs still alive after both signals."""
    if not pids:
        return []
    _send(pids, signal.SIGTERM)
    deadline = time.time() + term_wait
    while time.time() < deadline:
        time.sleep(0.1)
        alive = [p for p in pids if _pid_alive(p)]
        if not alive:
            return []
    survivors = [p for p in pids if _pid_alive(p)]
    if survivors:
        _send(survivors, signal.SIGKILL)
        time.sleep(0.5)
    return [p for p in survivors if _pid_alive(p)]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
