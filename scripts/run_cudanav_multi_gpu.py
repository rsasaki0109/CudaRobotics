#!/usr/bin/env python3
"""Run the deterministic CudaNav smoke on a matrix of visible GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from cudanav_multi_gpu import evaluate_multi_gpu_suite


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--devices",
        default="all",
        help="Comma-separated physical NVIDIA indices, or 'all'.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--minimum-gpu-devices", type=int, default=2)
    parser.add_argument("--minimum-gpu-models", type=int, default=2)
    return parser.parse_args()


def discover_gpus() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15.0,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"nvidia-smi GPU discovery failed: {error}") from error
    output = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            output.append(
                {
                    "index": fields[0],
                    "name": fields[1],
                    "uuid": fields[2],
                    "driver_version": fields[3],
                }
            )
    if not output:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return output


def output_is_ignored(path: Path) -> bool:
    if not path.is_relative_to(ROOT):
        return True
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    args = parse_args()
    if (
        args.repetitions <= 0
        or args.timeout_sec <= 0.0
        or args.minimum_gpu_devices <= 0
        or args.minimum_gpu_models <= 0
    ):
        raise SystemExit("matrix counts and timeout must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_dir}")
    if not output_is_ignored(output_dir):
        raise SystemExit(
            "output inside the repository must be git-ignored "
            "(use build/cudanav_multi_gpu/...)"
        )
    discovered = discover_gpus()
    if args.devices == "all":
        selected = discovered
    else:
        requested = {
            token.strip() for token in args.devices.split(",") if token.strip()
        }
        selected = [gpu for gpu in discovered if gpu["index"] in requested]
        missing = requested - {gpu["index"] for gpu in selected}
        if missing:
            raise SystemExit(f"unknown NVIDIA device indices: {sorted(missing)}")
    if not selected:
        raise SystemExit("no NVIDIA devices selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = {
        "schema_version": 1,
        "profile": "smoke",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "devices": selected,
        "repetitions": args.repetitions,
        "minimum_gpu_devices": args.minimum_gpu_devices,
        "minimum_gpu_models": args.minimum_gpu_models,
        "runs": [],
    }
    suite_path = output_dir / "multi_gpu_manifest.json"
    for gpu in selected:
        for repetition in range(args.repetitions):
            relative = Path(f"gpu_{gpu['index']}") / f"run_{repetition:02d}"
            run_directory = output_dir / relative
            driver_log = output_dir / f"gpu_{gpu['index']}_run_{repetition:02d}.log"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_cudanav_closed_loop.py"),
                "--output-dir",
                str(run_directory),
                "--profile",
                "smoke",
                "--timeout-sec",
                str(args.timeout_sec),
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu["index"]
            with driver_log.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            suite["runs"].append(
                {
                    "device": gpu,
                    "repetition": repetition,
                    "directory": relative.as_posix(),
                    "driver_log": driver_log.name,
                    "command": command,
                    "returncode": result.returncode,
                }
            )
            suite_path.write_text(
                json.dumps(suite, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    suite["finished_at"] = datetime.now(timezone.utc).isoformat()
    suite["gate"] = evaluate_multi_gpu_suite(suite, output_dir)
    suite["passed"] = suite["gate"]["passed"]
    suite_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(suite["gate"], indent=2, sort_keys=True))
    return 0 if suite["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
