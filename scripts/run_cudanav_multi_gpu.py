#!/usr/bin/env python3
"""Run the deterministic CudaNav smoke on a matrix of visible GPUs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cudanav_multi_gpu import evaluate_multi_gpu_suite
from cudanav_evidence import evaluate_manifest, evaluate_summary


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
    parser.add_argument(
        "--import-run",
        type=Path,
        action="append",
        default=[],
        help=(
            "Import a completed CudaNav smoke evidence directory from another "
            "machine. Repeat for each run; cannot be combined with local devices."
        ),
    )
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


def load_import(path: Path) -> tuple[dict, dict[str, str]]:
    source = path.resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing imported manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_gate = evaluate_manifest(manifest, source, "smoke")
    artifacts = manifest.get("artifacts", {})
    summary_path = (source / str(artifacts.get("summary", ""))).resolve()
    try:
        summary_path.relative_to(source)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_gate = evaluate_summary(summary, "smoke")
    except (OSError, ValueError, json.JSONDecodeError):
        summary_gate = {"passed": False, "checks": {"summary_readable": False}}
        summary = {}
    binding = (
        artifacts.get("trajectory") == summary.get("trajectory_csv")
        and manifest.get("traversal_count")
        == summary.get("traversals_requested")
    )
    if not (manifest_gate["passed"] and summary_gate["passed"] and binding):
        failed = sorted(
            [
                *(
                    f"manifest:{name}"
                    for name, passed in manifest_gate["checks"].items()
                    if not passed
                ),
                *(
                    f"summary:{name}"
                    for name, passed in summary_gate.get("checks", {}).items()
                    if not passed
                ),
                *(["artifact_binding"] if not binding else []),
            ]
        )
        raise ValueError(
            f"imported run failed smoke validation ({source}): "
            f"{', '.join(failed)}"
        )
    gpus = manifest.get("gpu")
    if not isinstance(gpus, list) or len(gpus) != 1:
        raise ValueError(f"imported run must bind exactly one GPU: {source}")
    gpu = gpus[0]
    required = ("physical_index", "name", "uuid")
    if not all(
        isinstance(gpu.get(field), str) and gpu[field] for field in required
    ):
        raise ValueError(f"imported run has incomplete GPU identity: {source}")
    return manifest, {
        "index": gpu["physical_index"],
        "name": gpu["name"],
        "uuid": gpu["uuid"],
        "driver_version": str(gpu.get("driver_version", "")),
    }


def import_suite(
    sources: list[Path],
    output_dir: Path,
    *,
    minimum_gpu_devices: int,
    minimum_gpu_models: int,
) -> dict:
    loaded = [(source.resolve(), *load_import(source)) for source in sources]
    by_uuid: dict[str, list[tuple[Path, dict, dict[str, str]]]] = {}
    for item in loaded:
        by_uuid.setdefault(item[2]["uuid"], []).append(item)
    repetition_counts = {len(items) for items in by_uuid.values()}
    if len(repetition_counts) != 1:
        raise ValueError(
            "every imported physical GPU must provide the same repetition count"
        )
    repetitions = repetition_counts.pop()
    devices = [items[0][2] for _, items in sorted(by_uuid.items())]
    suite = {
        "schema_version": 1,
        "profile": "smoke",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "collection_mode": "cross_machine_import",
        "devices": devices,
        "repetitions": repetitions,
        "minimum_gpu_devices": minimum_gpu_devices,
        "minimum_gpu_models": minimum_gpu_models,
        "runs": [],
    }
    for device_slot, (_, items) in enumerate(sorted(by_uuid.items())):
        for repetition, (source, manifest, device) in enumerate(
            sorted(items, key=lambda item: str(item[0]))
        ):
            relative = Path(f"gpu_{device_slot:02d}") / f"run_{repetition:02d}"
            destination = output_dir / relative
            shutil.copytree(source, destination)
            copied_manifest = destination / "manifest.json"
            suite["runs"].append(
                {
                    "device": device,
                    "repetition": repetition,
                    "directory": relative.as_posix(),
                    "driver_log": None,
                    "command": ["import", str(source)],
                    "returncode": 0,
                    "source_git_commit": manifest.get("git_commit"),
                    "source_config_sha256": manifest.get("config_sha256"),
                    "manifest_sha256": hashlib.sha256(
                        copied_manifest.read_bytes()
                    ).hexdigest(),
                }
            )
    return suite


def main() -> int:
    args = parse_args()
    if (
        args.repetitions <= 0
        or args.timeout_sec <= 0.0
        or args.minimum_gpu_devices <= 0
        or args.minimum_gpu_models <= 0
    ):
        raise SystemExit("matrix counts and timeout must be positive")
    if args.import_run and args.devices != "all":
        raise SystemExit("--import-run cannot be combined with --devices")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_dir}")
    if not output_is_ignored(output_dir):
        raise SystemExit(
            "output inside the repository must be git-ignored "
            "(use build/cudanav_multi_gpu/...)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = output_dir / "multi_gpu_manifest.json"
    if args.import_run:
        try:
            suite = import_suite(
                args.import_run,
                output_dir,
                minimum_gpu_devices=args.minimum_gpu_devices,
                minimum_gpu_models=args.minimum_gpu_models,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error
        suite["finished_at"] = datetime.now(timezone.utc).isoformat()
        suite["gate"] = evaluate_multi_gpu_suite(suite, output_dir)
        suite["passed"] = suite["gate"]["passed"]
        suite_path.write_text(
            json.dumps(suite, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(suite["gate"], indent=2, sort_keys=True))
        return 0 if suite["passed"] else 1

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
                    "manifest_sha256": (
                        hashlib.sha256(
                            (run_directory / "manifest.json").read_bytes()
                        ).hexdigest()
                        if (run_directory / "manifest.json").is_file()
                        else ""
                    ),
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
