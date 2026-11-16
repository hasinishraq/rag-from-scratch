"""One-command demo: builds the web UI if needed, then serves UI and API from one address.

Usage: python scripts/serve.py [--config configs/repliqa.yaml] [--port 8000] [--rebuild-ui]
For UI development with hot reload instead: npm --prefix web run dev (and run the API).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def build_ui(force: bool) -> bool:
    if (WEB / "dist" / "index.html").exists() and not force:
        return True
    npm = shutil.which("npm")
    if npm is None:
        print("Node.js (npm) is not installed, so the web UI cannot be built; the API still runs.")
        return False
    if not (WEB / "node_modules").exists():
        subprocess.run([npm, "install", "--no-fund", "--no-audit"], cwd=WEB, check=True)
    subprocess.run([npm, "run", "build"], cwd=WEB, check=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rebuild-ui", action="store_true", help="rebuild web/dist first")
    ap.add_argument("--config", default=None, help="config file (default: configs/default.yaml)")
    a = ap.parse_args()
    has_ui = build_ui(a.rebuild_ui)
    print(f"{'UI and API' if has_ui else 'API'} on http://{a.host}:{a.port}  (Ctrl+C stops)")
    # The app writes its own access log (with request ids), so uvicorn's is turned off.
    cmd = [sys.executable, "-m", "uvicorn", "rag.api:app", "--no-access-log"]
    cmd += ["--host", a.host, "--port", str(a.port)]
    env = {**os.environ, **({"RAG_CONFIG": a.config} if a.config else {})}
    try:
        return subprocess.call(cmd, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
