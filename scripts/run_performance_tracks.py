#!/usr/bin/env python3
"""Run paired GPU performance tracks, retaining failures and source provenance.

GPU runs are sequential so timing measurements do not contend with each other.
Render comparison videos separately with render_performance_tracks.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_results(output: str, track: str) -> list[dict]:
    rows = []
    for line in output.splitlines():
        if not line.startswith(f"RESULT {track} "):
            continue
        row = {}
        for item in line.split()[2:]:
            key, value = item.split("=", 1)
            if key == "mode":
                row[key] = value
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"Non-finite metric: {line}")
                row[key] = int(number) if number.is_integer() else number
        rows.append(row)
    if not rows:
        raise ValueError(f"Missing {track} result")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "build/performance_tracks")
    ap.add_argument("--bin-dir", type=Path)
    ap.add_argument("--dll-dir", type=Path, help="Optional OpenCV runtime directory on Windows")
    ap.add_argument("--dynamic-trials", type=int, default=60)
    ap.add_argument("--racing-seeds", type=int, default=10)
    ap.add_argument("--fleet-seeds", type=int, default=3)
    ap.add_argument("--ndt-tests", type=int, default=40)
    ap.add_argument("--seed-start", type=int, default=5000, help="Fresh dynamic/racing/fleet scenario seeds")
    ap.add_argument("--sequence", type=Path, default=ROOT / "build/datasets/mcd_ntu_day_02/loc_seq_v025.bin")
    ap.add_argument("--map-sequence", type=Path, default=ROOT / "build/datasets/mcd_ntu_night_13/loc_seq_s5_v025.bin")
    ap.add_argument("--skip-ndt", action="store_true", help="Explicitly skip if the external dataset is unavailable")
    args = ap.parse_args()
    if min(args.dynamic_trials, args.racing_seeds, args.fleet_seeds, args.ndt_tests) < 1:
        ap.error("trial counts must be positive")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bin_dir = args.bin_dir or ROOT / ("bin/Release" if os.name == "nt" else "bin")
    env = os.environ.copy()
    if args.dll_dir:
        env["PATH"] = str(args.dll_dir.resolve()) + os.pathsep + env.get("PATH", "")
    commands = []
    evidence = {"commands": commands, "tracks": {}, "sources": {}, "binaries": {}, "datasets": {}}
    source_files = ["src/benchmark_ndt_localization.cu", "src/gpu_esdf_mppi_3d.cu", "src/gpu_mppi_racing.cu",
                    "src/gpu_fleet_traffic.cu", "include/mppi_reduction.cuh", "scripts/run_performance_tracks.py"]
    evidence["sources"] = {p: digest(ROOT / p) for p in source_files}
    evidence["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    evidence["worktree_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True))
    evidence["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], text=True).strip()

    def save():
        (out / "results.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def run(target: str, options: list[str], name: str) -> str:
        binary = (bin_dir / (target + (".exe" if os.name == "nt" else ""))).resolve()
        evidence["binaries"].setdefault(target, digest(binary))
        command = [str(binary), *options]
        print(f"Running {name}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        (out / f"{name}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        commands.append({"name": name, "argv": command, "returncode": completed.returncode})
        save()
        if completed.returncode != 0:
            raise RuntimeError(f"{name} exited {completed.returncode}; see retained log")
        return completed.stdout

    dyn = run("gpu_esdf_mppi_3d", ["--movers", "6", "--mover-speed", "3", "--trials", str(args.dynamic_trials),
              "--observed-trials", "--observation-delay", "3", "--observation-period", "2", "--turn-period", "20",
              "--seed", str(args.seed_start), "--no-video", "--headless"], "dynamic")
    dynamic_rows = parse_results(dyn, "dynamic")
    if len(dynamic_rows) != args.dynamic_trials * 3:
        raise ValueError("Incomplete dynamic matrix")
    evidence["tracks"]["dynamic"] = dynamic_rows
    for track, target, count in [("racing", "gpu_mppi_racing", args.racing_seeds), ("fleet", "gpu_fleet_traffic", args.fleet_seeds)]:
        evidence["tracks"][track] = []
        for seed in range(args.seed_start, args.seed_start + count):
            for improved in [False, True]:
                options = ["--seed", str(seed), "--no-video", "--headless"]
                if track == "racing":
                    options += ["--grip-plant", "--laps", "2", "--steps", "1800"]
                    if improved:
                        options += ["--grip-aware"]
                else:
                    options += ["--robots", "200", "--steps", "1800"]
                    if improved:
                        options += ["--platoon"]
                rows = parse_results(run(target, options, f"{track}_{seed}_{'improved' if improved else 'baseline'}"), track)
                if len(rows) != 1 or rows[0]["seed"] != seed:
                    raise ValueError(f"Invalid {track} result binding")
                evidence["tracks"][track].extend(rows)
    if not args.skip_ndt:
        for name, path in [("sequence", args.sequence), ("map_sequence", args.map_sequence)]:
            evidence["datasets"][name] = {"path": str(path.resolve()), "sha256": digest(path), "bytes": path.stat().st_size}
        evidence["tracks"]["ndt"] = []
        for seed in [1, 41]:
            for improved in [False, True]:
                mode = "cascade" if improved else "full"
                csv_path = out / f"ndt_{mode}_{seed}.csv"
                options = ["--sequence", str(args.sequence.resolve()), "--map-sequence", str(args.map_sequence.resolve()),
                           "--prior-err", "10", "--grid", "11", "--grid-step", "2", "--yaws", "16", "--tests", str(args.ndt_tests),
                           "--cpu-tests", "0", "--seed", str(seed), "--csv", str(csv_path)]
                if improved:
                    options += ["--screen-stride", "4", "--screen-iters", "12", "--survivors", "64"]
                run("benchmark_ndt_localization", options, f"ndt_{mode}_{seed}")
                with csv_path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if len(rows) != args.ndt_tests:
                    raise ValueError("Incomplete NDT matrix")
                for row in rows:
                    evidence["tracks"]["ndt"].append({"mode": mode, "seed": seed, **{k: float(v) for k, v in row.items()}})
    else:
        evidence["ndt_skipped"] = True
    save()
    table = ["# Paired GPU performance tracks", "", f"GPU: {evidence['gpu']}", "",
             "All cases, including failures, are retained in results.json and logs. GPU runs execute sequentially.", "",
             "| Track / method | Outcome | Mean compute ms | Mean per-run p95 ms |", "|---|---|---:|---:|"]
    for track, rows in evidence["tracks"].items():
        for mode in dict.fromkeys(r["mode"] for r in rows):
            group = [r for r in rows if r["mode"] == mode]
            if track == "dynamic":
                outcome = f"{sum(r['success'] for r in group)}/{len(group)} success; {sum(r['collision'] for r in group)} collisions"
            elif track == "racing":
                outcome = f"{sum(r['laps'] == r['target'] and r['offtrack'] == 0 for r in group)}/{len(group)} clean 2-lap runs"
            elif track == "fleet":
                outcome = f"{statistics.mean(r['arrived'] for r in group):.1f}/200 delivered in 180 s; {sum(r['collisions'] for r in group)} collisions"
            else:
                outcome = f"{sum(r['init_ok'] for r in group):.0f}/{len(group)} recovered"
            mean = statistics.mean(r["gpu_ms"] if track == "ndt" else r["mean_ms"] for r in group)
            p95 = "n/a" if track == "ndt" else f"{statistics.mean(r['p95_ms'] for r in group):.3f}"
            table.append(f"| {track} / {mode} | {outcome} | {mean:.3f} | {p95} |")
    table += ["", "Scope: synthetic position observations with stable object IDs; known racing surface map and a friction-limited bicycle approximation; four straight fleet lanes and a serial reservation baseline. NDT uses one campus sequence and another session's map, with gravity/z priors. These are development benchmarks, not release or deployment attestations."]
    (out / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
