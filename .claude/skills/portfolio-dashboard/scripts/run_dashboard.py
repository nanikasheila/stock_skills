#!/usr/bin/env python3
"""ポートフォリオダッシュボード — ランチャースクリプト.

Streamlit サーバーを起動してブラウザでダッシュボードを表示する。

Usage
-----
    python3 .claude/skills/portfolio-dashboard/scripts/run_dashboard.py [--port PORT]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def _find_python() -> str:
    """venv の Python を優先的に検出する."""
    project_root = Path(__file__).resolve().parents[4]
    # Windows venv
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    # Linux/Mac venv
    venv_python_unix = project_root / ".venv" / "bin" / "python"
    if venv_python_unix.exists():
        return str(venv_python_unix)
    # fallback to current interpreter
    return sys.executable


def main():
    parser = argparse.ArgumentParser(description="Portfolio Dashboard Launcher")
    parser.add_argument(
        "--port", type=int, default=8501,
        help="Streamlit サーバーのポート番号 (default: 8501)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="ブラウザを自動で開かない",
    )
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parent / "app.py"
    if not app_path.exists():
        print(f"Error: {app_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    python_exe = _find_python()
    url = f"http://localhost:{args.port}"

    cmd = [
        python_exe, "-m", "streamlit", "run",
        str(app_path),
        f"--server.port={args.port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    print(f"🚀 ダッシュボードを起動中... → {url}")
    print(f"   Python: {python_exe}")
    print(f"   停止するには Ctrl+C を押してください")
    print()

    project_root = str(Path(__file__).resolve().parents[4])

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(app_path.parent),
            env={**os.environ, "PYTHONPATH": project_root},
        )

        if not args.no_browser:
            time.sleep(3)
            webbrowser.open(url)

        proc.wait()
    except KeyboardInterrupt:
        print("\n⏹️  ダッシュボードを停止しました")
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
