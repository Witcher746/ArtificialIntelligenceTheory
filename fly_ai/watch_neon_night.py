"""Restart the crash-resumable Neon Drive trainer for a bounded night run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .train_neon_night import atomic_json, prevent_windows_sleep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--output-dir", default="artifacts/fly_ai/neon_night")
    parser.add_argument("--max-restarts", type=int, default=5)
    args = parser.parse_args()
    if args.hours <= 0:
        raise SystemExit("hours must be positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    watchdog_path = output / "watchdog.json"
    stdout_path = output.parent / f"{output.name}_stdout.log"
    stderr_path = output.parent / f"{output.name}_stderr.log"
    deadline = time.monotonic() + args.hours * 3600.0
    restarts = 0
    prevent_windows_sleep(True)
    try:
        while time.monotonic() < deadline and not (output / "STOP").exists():
            remaining_hours = max(0.001, (deadline - time.monotonic()) / 3600.0)
            command = [
                sys.executable,
                "-m",
                "fly_ai.train_neon_night",
                "--hours",
                f"{remaining_hours:.6f}",
                "--output-dir",
                str(output),
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
                stdout.write(f"\nWATCHDOG START {time.strftime('%Y-%m-%d %H:%M:%S')} remaining={remaining_hours:.3f}h\n")
                stdout.flush()
                process = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=flags,
                )
                atomic_json(watchdog_path, {
                    "state": "RUNNING",
                    "watchdog_pid": os.getpid(),
                    "trainer_pid": process.pid,
                    "restarts": restarts,
                    "remaining_hours": remaining_hours,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                return_code = process.wait()
            if return_code == 0:
                atomic_json(watchdog_path, {
                    "state": "COMPLETE",
                    "watchdog_pid": os.getpid(),
                    "trainer_pid": process.pid,
                    "restarts": restarts,
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                return
            restarts += 1
            atomic_json(watchdog_path, {
                "state": "RESTARTING" if restarts <= args.max_restarts else "FAILED",
                "watchdog_pid": os.getpid(),
                "trainer_pid": process.pid,
                "return_code": return_code,
                "restarts": restarts,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            if restarts > args.max_restarts:
                raise SystemExit(f"trainer failed {restarts} times; see {stderr_path}")
            time.sleep(15.0)
    finally:
        prevent_windows_sleep(False)


if __name__ == "__main__":
    main()
