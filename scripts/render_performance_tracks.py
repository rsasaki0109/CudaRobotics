#!/usr/bin/env python3
"""Render measured performance charts and paired demo GIFs from retained runs.

Requires numpy, matplotlib, opencv-python and Pillow. NDT animation shows
initial/final pose snapshots; optimizer iterations are not fabricated.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def paired_video(left: Path, right: Path, out: Path, step_seconds: float, skip: int, titles: tuple[str, str],
                 outcomes: tuple[str, str], limits: tuple[int, int] | None = None) -> None:
    caps = [cv2.VideoCapture(str(p)) for p in (left, right)]
    if not all(c.isOpened() for c in caps):
        raise RuntimeError(f"Cannot load comparison videos: {left}, {right}")
    frames, last = [], [None, None]
    index = 0
    while True:
        reads = [c.read() if limits is None or index < limits[i] else (False, None) for i, c in enumerate(caps)]
        if not any(ok for ok, _ in reads):
            break
        for i, (ok, frame) in enumerate(reads):
            if ok:
                last[i] = frame
        if index % skip == 0 and all(f is not None for f in last):
            panels = []
            for i, (title, frame, outcome) in enumerate(zip(titles, last, outcomes)):
                if frame.shape[1] == 1040:  # Fleet: magnify the existing intersection view.
                    frame = frame.copy()
                    frame[160:340, 850:1030] = cv2.resize(frame[350:490, 350:490], (180, 180))
                    cv2.putText(frame, "Intersection", (850, 145), cv2.FONT_HERSHEY_SIMPLEX, .65, (220, 230, 240), 1)
                scale = 540 / frame.shape[1]
                panel = cv2.resize(frame, (540, int(frame.shape[0] * scale)))
                banner = np.full((94, 540, 3), (25, 30, 36), np.uint8)
                cv2.putText(banner, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(banner, f"simulation t={index * step_seconds:.1f}s; playback x{skip * step_seconds / .10:.1f}",
                            (12, 47), cv2.FONT_HERSHEY_SIMPLEX, .43, (190, 205, 220), 1, cv2.LINE_AA)
                cv2.putText(banner, "Run result: " + outcome, (12, 77), cv2.FONT_HERSHEY_SIMPLEX, .56,
                            (120, 165, 245) if i == 0 else (135, 230, 110), 1, cv2.LINE_AA)
                panels.append(np.vstack([banner, panel]))
            height = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, height-p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(25, 30, 36)) for p in panels]
            frames.append(Image.fromarray(cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB)))
        index += 1
    for c in caps:
        c.release()
    if not frames:
        raise ValueError("Empty comparison")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False)


def rotation(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def sequence_clouds(path: Path, wanted: int | None = None):
    clouds = []
    with path.open("rb") as f:
        magic, version, count = struct.unpack("<8sII", f.read(16))
        if not magic.startswith(b"CRLOC1") or version != 1:
            raise ValueError("Unexpected localization sequence")
        for i in range(count):
            stamp, *pose, n = struct.unpack("<d7dI", f.read(68))
            if wanted is not None and i != wanted:
                f.seek(n * 12, 1)
                continue
            if wanted is None and i % 5:
                f.seek(n * 12, 1)
                continue
            points = np.frombuffer(f.read(n * 12), dtype="<f4").reshape(-1, 3).copy()
            points = points[::max(1, len(points)//2000)]
            clouds.append((np.array(pose), points))
            if wanted is not None:
                break
    if not clouds:
        raise ValueError(f"No requested scan in {path}")
    return clouds


def ndt_animation(evidence: dict, out: Path) -> None:
    rows = evidence["tracks"]["ndt"]
    full = next(r for r in rows if r["mode"] == "full" and r["seed"] == 1)
    cascade = next(r for r in rows if r["mode"] == "cascade" and r["seed"] == 1 and r["frame"] == full["frame"])
    sequence = Path(evidence["datasets"]["sequence"]["path"])
    origin = sequence_clouds(sequence, 0)[0][0][:3]
    pose, points = sequence_clouds(sequence, int(full["frame"]))[0]
    map_clouds = sequence_clouds(Path(evidence["datasets"]["map_sequence"]["path"]))
    mapped = np.vstack([p @ rotation(q[3:7]).T + q[:3] - origin for q, p in map_clouds])
    centre = pose[:2] - origin[:2]

    def paint(row: dict, locked: bool) -> np.ndarray:
        img = np.full((570, 570, 3), (24, 28, 32), np.uint8)
        def scatter(cloud, color):
            xy = (cloud[:, :2]-centre) * 8.0 + 285
            xy[:, 1] = 570-xy[:, 1]
            mask = np.isfinite(xy).all(axis=1) & (xy >= 0).all(axis=1) & (xy < 570).all(axis=1)
            xy = xy[mask].astype(int)
            img[xy[:, 1], xy[:, 0]] = color
        scatter(mapped, (105, 115, 120))
        if locked:
            R = np.array([row[f"est_r{i}{j}"] for i in range(3) for j in range(3)]).reshape(3, 3)
            t = np.array([row[f"est_t{axis}"] for axis in "xyz"])
        else:
            phase = row["prior_phase"]
            Rz = np.array([[np.cos(phase), -np.sin(phase), 0], [np.sin(phase), np.cos(phase), 0], [0, 0, 1]])
            R = Rz @ rotation(pose[3:7])
            t = np.array([row[f"prior_{axis}"] for axis in "xyz"])
        scatter(points @ R.T + t, (90, 245, 110) if locked else (80, 100, 255))
        cv2.putText(img, "ALIGNED" if locked else "UNCERTAIN INITIAL POSE", (16, 545), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (90, 245, 110) if locked else (80, 100, 255), 2, cv2.LINE_AA)
        return img
    snapshots = [[paint(row, state) for state in [False, True]] for row in [full, cascade]]
    frames = []
    for i in range(55):
        elapsed_ms = max(0, i-5) * 10  # 100 ms/frame: measured time slowed down 10x.
        panels = []
        for j, (row, label) in enumerate([(full, "FULL SEARCH"), (cascade, "STAGED SEARCH")]):
            banner = np.full((70, 570, 3), (24, 28, 32), np.uint8)
            cv2.putText(banner, f"{label}: measured {row['gpu_ms']:.1f} ms", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (240, 240, 240), 1)
            cv2.putText(banner, f"pose snapshots; timing slowed 10x; t={elapsed_ms} ms", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .44, (190, 205, 220), 1)
            panels.append(np.vstack([banner, snapshots[j][int(elapsed_ms >= row['gpu_ms'])]]))
        frames.append(Image.fromarray(cv2.cvtColor(np.hstack(panels), cv2.COLOR_BGR2RGB)))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--videos", action="store_true", help="Run selected paired examples and render GIFs")
    ap.add_argument("--dll-dir", type=Path)
    args = ap.parse_args()
    evidence = json.loads(args.results.read_text(encoding="utf-8"))
    out = args.results.parent / "media"; out.mkdir(exist_ok=True)
    tracks = evidence["tracks"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor="#f6f8fb")
    plots = [
        ("dynamic", ["observed_stale", "observed_risk"], "Delayed-observation navigation", "collision", "collision episodes"),
        ("ndt", ["full", "cascade"], "Real-data localization", "gpu_ms", "ms"),
        ("fleet", ["serial", "platoon"], "200-robot intersection", "arrived", "robots / 180 s"),
        ("racing", ["blind", "aware"], "Friction-limited racing", "laps", "rate"),
    ]
    for ax, (track, modes, title, metric, unit) in zip(axes.flat, plots):
        if track not in tracks:
            ax.set_visible(False); continue
        groups = [[r for r in tracks[track] if r["mode"] == mode] for mode in modes]
        values = []
        for g in groups:
            if track == "racing":
                values.append(statistics.mean(r["laps"] == r["target"] and r["offtrack"] == 0 for r in g) * 100)
            elif track == "dynamic":
                values.append(sum(r[metric] for r in g))
            elif unit == "rate":
                values.append(statistics.mean(r[metric] for r in g) * 100)
            else:
                values.append(statistics.mean(r[metric] for r in g))
        bars = ax.bar(["Baseline", "Improved"], values, color=["#d16f69", "#258c80"], width=.55)
        ax.bar_label(bars, labels=[f"{v:.1f}{'%' if unit == 'rate' else ''}" for v in values], padding=5)
        ax.set_title(title, loc="left", fontweight="bold"); ax.set_ylabel("success %" if unit == "rate" else unit)
        ax.set_ylim(0, max(max(values) * 1.25, 1)); ax.spines[["top", "right"]].set_visible(False)
        ax.text(.02, .97, f"{len(groups[0])} paired cases", transform=ax.transAxes, va="top", fontsize=9, color="#586270")
        if track == "dynamic":
            ax.text(.98, .88, "Goals: " + " / ".join(str(sum(r['success'] for r in g)) for g in groups), transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.suptitle("CudaRobotics: measured performance improvements", fontsize=17, fontweight="bold")
    fig.text(.02, .02, "Sequential GPU measurements. Full results include failures. Synthetic navigation/fleet/racing; real-data NDT.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94)); fig.savefig(out / "performance_comparison.png", dpi=160); plt.close(fig)
    if "ndt" in tracks:
        ndt_animation(evidence, out / "ndt_comparison.gif")
    if args.videos:
        capture_dir = out / "captures"; capture_dir.mkdir(exist_ok=True)
        capture_gif = capture_dir / "gif"
        env = os.environ.copy()
        if args.dll_dir:
            env["PATH"] = str(args.dll_dir.resolve()) + os.pathsep + env.get("PATH", "")
        captures = []
        def run(target, options, name):
            binary = ROOT / ("bin/Release" if os.name == "nt" else "bin") / (target + (".exe" if os.name == "nt" else ""))
            command = [str(binary), *options, "--headless"]
            result = subprocess.run(command, cwd=capture_dir, env=env, capture_output=True, text=True)
            (out / f"{name}.log").write_text(result.stdout+result.stderr, encoding="utf-8")
            captures.append({"name": name, "argv": command, "returncode": result.returncode})
            if result.returncode not in [0, 1] or "RESULT " not in result.stdout:
                raise RuntimeError(f"Failed capture {name}")
            line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
            return dict(item.split("=", 1) for item in line.split()[2:])
        # First paired improvement in ascending seed order, explicitly retained.
        dynamic = tracks["dynamic"]
        seeds = sorted({r["seed"] for r in dynamic})
        selected = next((s for s in seeds if any(r["seed"] == s and r["mode"] == "observed_stale" and r["collision"] for r in dynamic)
                         and any(r["seed"] == s and r["mode"] == "observed_risk" and r["success"] for r in dynamic)), seeds[0])
        dynamic_captures = []
        for mode in [5, 7]:
            dynamic_captures.append(run("gpu_esdf_mppi_3d", ["--movers", "6", "--mover-speed", "3", "--observation-delay", "3", "--turn-period", "20",
                "--seed", str(selected), "--mode", str(mode)], f"dynamic_{mode}"))
        paired_video(capture_gif/"gpu_esdf_mppi_3d_observed_stale.avi", capture_gif/"gpu_esdf_mppi_3d_observed_risk.avi",
                     out/"dynamic_comparison.gif", .1, 2, (f"Stale observation; selected seed {selected}", "Latency + envelope + feasible selection"),
                     tuple("collision" if int(r['collision']) else "goal reached" if int(r['success']) else "timeout" for r in dynamic_captures),
                     tuple(int(r['steps']) for r in dynamic_captures))
        example_seed = min(r['seed'] for r in tracks['racing'])
        race_captures, fleet_captures = [], []
        for improved in [False, True]:
            race_captures.append(run("gpu_mppi_racing", ["--grip-plant", "--seed", str(example_seed), "--laps", "2", "--steps", "1800"] + (["--grip-aware"] if improved else []), f"racing_{improved}"))
            fleet_captures.append(run("gpu_fleet_traffic", ["--seed", str(example_seed), "--steps", "1800"] + (["--platoon"] if improved else []), f"fleet_{improved}"))
        paired_video(capture_gif/"gpu_mppi_racing_grip_blind.avi", capture_gif/"gpu_mppi_racing_grip_aware.avi",
                     out/"racing_comparison.gif", .18, 2, ("Kinematic proposals on slippery surface", "Friction-aware feasible proposals"),
                     tuple(f"{r['laps']}/2 laps; {r['offtrack']} off-track steps" for r in race_captures))
        paired_video(capture_gif/"gpu_fleet_traffic_serial.avi", capture_gif/"gpu_fleet_traffic_platoon.avi",
                     out/"fleet_comparison.gif", .2, 6, ("One reservation per robot", "Compatible platoon reservations"),
                     tuple(f"{r['arrived']}/200 delivered in 180s; {r['collisions']} contacts" for r in fleet_captures))
        (out/"captures.json").write_text(json.dumps(captures, indent=2)+"\n", encoding="utf-8")
    print(f"Rendered comparisons: {out}")


if __name__ == "__main__":
    main()
