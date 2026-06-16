"""brewctl check — running llama-server processes + their port/RAM/model."""
from __future__ import annotations

import logging
import re
import subprocess

from ._proc import pgrep_pids

logger = logging.getLogger(__name__)

PORT_RE = re.compile(r"--port\s+(\d+)")
MODEL_RE = re.compile(r"(models/\S+)")


def _ps_command(pid: int) -> str:
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip()
    except FileNotFoundError:
        return ""


def _ps_rss_mb(pid: int) -> int:
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "rss="],
            capture_output=True, text=True, check=False,
        )
        kb = int(out.stdout.strip() or "0")
        return kb // 1024
    except (FileNotFoundError, ValueError):
        return 0


def run_check() -> int:
    pids = pgrep_pids("llama-server")
    print("--- MATRIX SWARM REAL-TIME STATUS ---")
    print(f"{'PID':<8} {'PORT':<8} {'RAM (MB)':<10} {'MODEL':<60}")
    print("-" * 86)
    for pid in pids:
        cmd = _ps_command(pid)
        port_m = PORT_RE.search(cmd)
        model_m = MODEL_RE.search(cmd)
        port = port_m.group(1) if port_m else "-"
        model = model_m.group(1) if model_m else "-"
        rss = _ps_rss_mb(pid)
        print(f"{pid:<8} {port:<8} {rss:<10} {model:<60}")
    print("-" * 86)
    print(f"{len(pids)} llama-server process(es) running")
    return 0
