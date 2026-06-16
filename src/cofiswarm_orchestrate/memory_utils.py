"""Cross-platform host memory and backend selection for Python orchestration (MS-25).

Mirrors cpp_core/src/host_memory.cpp JSON shape for orchestration-side caps.
Stdlib-only — no psutil dependency.
"""
from __future__ import annotations

import logging
import platform
import re
import shutil
import socket
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Conservative fallback when detection fails (Windows / unknown).
_FALLBACK_AVAILABLE_GB = 16.0

_MEMINFO_KB = re.compile(r"^(\w+):\s+(\d+)\s+kB$")
_DARWIN_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_DARWIN_FREE_RE = re.compile(
    r"Pages free:\s+(\d+).*?"
    r"Pages inactive:\s+(\d+).*?"
    r"Pages speculative:\s+(\d+)",
    re.DOTALL,
)


def _round_gb(bytes_val: float) -> float:
    return round(bytes_val / (1024**3), 1)


def _linux_snapshot() -> dict[str, Any]:
    total_kb = avail_kb = free_kb = buffers_kb = cached_kb = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                m = _MEMINFO_KB.match(line.strip())
                if not m:
                    continue
                key, kb = m.group(1), int(m.group(2))
                if key == "MemTotal":
                    total_kb = kb
                elif key == "MemAvailable":
                    avail_kb = kb
                elif key == "MemFree":
                    free_kb = kb
                elif key == "Buffers":
                    buffers_kb = kb
                elif key == "Cached":
                    cached_kb = kb
    except OSError as exc:
        logger.error("linux meminfo read failed: %s", exc)
        return {"ok": False}

    if total_kb <= 0:
        return {"ok": False}

    if avail_kb <= 0:
        avail_kb = free_kb + buffers_kb + cached_kb

    total = total_kb * 1024
    free = avail_kb * 1024
    used = max(0, total - free)
    return {
        "ok": True,
        "total_gb": _round_gb(total),
        "used_gb": _round_gb(used),
        "free_gb": _round_gb(free),
    }


def _darwin_snapshot() -> dict[str, Any]:
    try:
        total_out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, check=False,
        )
        total_bytes = int((total_out.stdout or "").strip() or "0")
        if total_bytes <= 0:
            return {"ok": False}

        vm_out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=False,
        )
        vm_text = vm_out.stdout or ""
        page_m = _DARWIN_PAGE_RE.search(vm_text)
        page_size = int(page_m.group(1)) if page_m else 4096

        free_m = _DARWIN_FREE_RE.search(vm_text)
        if not free_m:
            return {"ok": False}

        pages = sum(int(free_m.group(i)) for i in range(1, 4))
        free = pages * page_size
        used = max(0, total_bytes - free)
        return {
            "ok": True,
            "total_gb": _round_gb(total_bytes),
            "used_gb": _round_gb(used),
            "free_gb": _round_gb(free),
        }
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.error("darwin memory snapshot failed: %s", exc)
        return {"ok": False}


def get_host_memory_snapshot() -> dict[str, Any]:
    """Return host RAM snapshot matching GET /api/memory JSON shape."""
    sys_name = platform.system()
    out: dict[str, Any] = {
        "ok": False,
        "source": "host",
        "platform": "darwin" if sys_name == "Darwin" else sys_name.lower(),
        "total_gb": None,
        "used_gb": None,
        "free_gb": None,
    }

    if sys_name == "Darwin":
        out["platform"] = "darwin"
        snap = _darwin_snapshot()
    elif sys_name == "Linux":
        out["platform"] = "linux"
        snap = _linux_snapshot()
    else:
        out["platform"] = "unknown"
        return out

    if not snap.get("ok"):
        return out

    out.update(snap)
    return out


def get_available_memory_gb() -> float:
    """Available host RAM in GB for orchestration memory-cap checks."""
    snap = get_host_memory_snapshot()
    if snap.get("ok") and snap.get("free_gb") is not None:
        return float(snap["free_gb"])
    logger.warning(
        "host memory unavailable on %s — using %.1f GB fallback",
        platform.system(), _FALLBACK_AVAILABLE_GB,
    )
    return _FALLBACK_AVAILABLE_GB


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def docker_vllm_reachable(ports: tuple[int, ...] = (8080, 8081, 8082, 8083)) -> bool:
    """True when at least one vLLM port responds (Docker Model Runner)."""
    return any(_port_open("127.0.0.1", p) for p in ports)


def preferred_backend() -> str:
    """Platform-aware default inference backend for Python orchestration modes."""
    sys_name = platform.system()
    if sys_name == "Darwin":
        return "mlx"
    if sys_name == "Linux":
        if shutil.which("nvidia-smi") and docker_vllm_reachable():
            return "vllm"
        if shutil.which("llama-server"):
            return "llama.cpp"
        return "vllm"
    if shutil.which("llama-server"):
        return "llama.cpp"
    return "llama.cpp"


def mode_memory_weight_scale() -> float:
    """Adjust per-mode memoryWeight on non-MLX platforms (vLLM VRAM overhead)."""
    if platform.system() == "Darwin":
        return 1.0
    backend = preferred_backend()
    if backend == "vllm":
        return 1.15
    return 1.0


# Mirror src/utils/modeManifestData.js python-mode memoryWeight values.
_MODE_MEMORY_WEIGHT: dict[str, float] = {
    "map_reduce": 3.0,
    "speculative": 2.0,
    "critic_debate": 2.0,
    "tree_of_thought": 3.0,
}
_GB_PER_WEIGHT = 4.0


def required_memory_gb(mode_id: str) -> float:
    """Host RAM (GB) required before starting a Python orchestration mode."""
    weight = _MODE_MEMORY_WEIGHT.get(mode_id, 2.0)
    return weight * _GB_PER_WEIGHT * mode_memory_weight_scale()


def check_mode_memory_ok(mode_id: str) -> tuple[bool, str | None]:
    """Return (ok, error_message). Uses live host snapshot with conservative fallback."""
    required = required_memory_gb(mode_id)
    free = get_available_memory_gb()
    if free < required:
        return False, (
            f"insufficient host memory for {mode_id!r}: "
            f"{free:.1f} GB free, ~{required:.1f} GB required"
        )
    return True, None
