#!/usr/bin/env python3
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("ORCH_SIDECAR_PORT", "3003")
from cofiswarm_orchestrate.mlx_coordinator.sidecar import main
if __name__ == "__main__":
    main()
