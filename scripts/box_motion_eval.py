#!/usr/bin/env python3
"""Tracking moving traffic in gpu_ground_segmentation (--moving), dev vs held-out.

--moving adds a lead car and a following car in the sensor's lane to the drive
of --sequence. Two trackers run side by side: the static one (world-frame
accumulation) and the motion one (constant-velocity Kalman filter, accumulation
in the object's frame, a stand-still vs moving test per refit). This script runs
the dev drive (seed 0) and held-out drives (seeds 1..N) and compares, on the
moving and on the parked vehicles, the boxes with the size prior of: a single
scan, the static tracker and the motion tracker; plus velocity errors and
identity switches. Exact sign tests per observation and per seed.

Writes <out>.csv (all vehicle observations) and <out>.md (the report).
"""

import argparse
import csv
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from box_fitting_heldout import ROOT, find_binary, sign_test  # noqa: E402

VEHICLES = {"0", "1", "2", "4", "7", "8"}
METHODS = [("prior_mlp", "single scan"), ("trk_prior", "static tracker"), ("mtrk_prior", "motion tracker")]
COMPARISONS = [  # (group, a, b, metric, lower is better)
    ("moving", "mtrk_prior", "trk_prior", "iou", False),
    ("moving", "mtrk_prior", "trk_prior", "centre", True),
    ("moving", "mtrk_prior", "prior_mlp", "iou", False),
    ("moving", "mtrk_prior", "prior_mlp", "centre", True),
    ("parked", "mtrk_prior", "trk_prior", "iou", False),
    ("parked", "mtrk_prior", "prior_mlp", "iou", False),
]
LABEL = dict(METHODS)
METRIC = {"iou": "BEV IoU", "centre": "centre error"}


def run(binary, seed, tmp_csv):
    subprocess.run([binary, "--no-video", "--moving", "--seed", str(seed), "--obs-csv", tmp_csv],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(tmp_csv, newline="") as f:
        return [r for r in csv.DictReader(f) if r["box"] in VEHICLES]


def group(rows, g):
    return [r for r in rows if (float(r["speed"]) > 0) == (g == "moving")]


def switches(rows, col):
    n, last = 0, {}
    for r in rows:
        key, t = (r["seed"], r["box"]), int(r[col])
        if t < 0:
            continue
        if key in last and last[key] != t:
            n += 1
        last[key] = t
    return n


def verr(rows, col):
    v = [float(r[col]) for r in rows if float(r["verr_static"]) >= 0 and float(r["verr_motion"]) >= 0]
    return sum(v) / len(v) if v else float("nan"), len(v)


def compare(rows, a, b, metric, lower):
    w = l = t = 0
    for r in rows:
        d = float(r["%s_%s" % (metric, a)]) - float(r["%s_%s" % (metric, b)])
        if abs(d) < 1e-9:
            t += 1
        elif (d < 0) == lower:
            w += 1
        else:
            l += 1
    sw = sl = 0
    for sd in sorted({r["seed"] for r in rows}):
        d = sum(float(r["%s_%s" % (metric, a)]) - float(r["%s_%s" % (metric, b)]) for r in rows if r["seed"] == sd)
        if abs(d) < 1e-9:
            continue
        if (d < 0) == lower:
            sw += 1
        else:
            sl += 1
    diff = sum(float(r["%s_%s" % (metric, a)]) - float(r["%s_%s" % (metric, b)]) for r in rows) / max(1, len(rows))
    return diff, w, l, t, sign_test(w, l), sw, sl, sign_test(sw, sl)


def section(w, name, rows):
    w.write("\n## %s\n\n" % name)
    w.write("| Vehicles | observations | box (with the size prior) | BEV IoU | IoU >= 0.5 | centre error |\n"
            "|---|---:|---|---:|---:|---:|\n")
    for g in ("moving", "parked"):
        gr = group(rows, g)
        for key, label in METHODS:
            iou = [float(r["iou_" + key]) for r in gr]
            ctr = [float(r["centre_" + key]) for r in gr]
            w.write("| %s | %d | %s | %.3f | %d | %.2f m |\n" % (g, len(gr), label, sum(iou) / max(1, len(iou)),
                                                               sum(v >= 0.5 for v in iou), sum(ctr) / max(1, len(ctr))))
    w.write("\n| Vehicles | velocity error, static tracker | velocity error, motion tracker | observations |\n"
            "|---|---:|---:|---:|\n")
    for g in ("moving", "parked"):
        vs, n = verr(group(rows, g), "verr_static")
        vm, _ = verr(group(rows, g), "verr_motion")
        w.write("| %s | %.2f m/s | %.2f m/s | %d |\n" % (g, vs, vm, n))
    w.write("\nIdentity switches: static tracker %d, motion tracker %d.\n"
            % (switches(rows, "track"), switches(rows, "mtrack")))
    multi = len({r["seed"] for r in rows}) > 1
    w.write("\n| Vehicles | comparison | metric | mean difference | observations better / worse / tie | sign test p |")
    w.write(" seeds better / worse | seed sign test p |\n" if multi else "\n")
    w.write("|---|---|---|---:|---:|---:|" + ("---:|---:|\n" if multi else "\n"))
    for g, a, b, metric, lower in COMPARISONS:
        diff, wn, ls, t, p, sw, sl, sp = compare(group(rows, g), a, b, metric, lower)
        w.write("| %s | %s vs %s | %s | %+.3f | %d / %d / %d | %.2g |" % (g, LABEL[a], LABEL[b], METRIC[metric],
                                                                       diff, wn, ls, t, p))
        w.write(" %d / %d | %.2g |\n" % (sw, sl, sp) if multi else "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=20, help="held-out seeds 1..N")
    ap.add_argument("--out", default="docs/results/box_motion_2026-10-05")
    args = ap.parse_args()
    binary = find_binary()
    tmp = os.path.join(ROOT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    tmp_csv = os.path.join(tmp, "box_motion_obs.csv")
    dev = run(binary, 0, tmp_csv)
    held = []
    for seed in range(1, args.seeds + 1):
        held += run(binary, seed, tmp_csv)
        print("seed %d done" % seed)
    w = io.StringIO()
    w.write("# LiDAR box tracking with moving traffic: dev drive vs held-out drives\n\n")
    w.write("Generated by `scripts/box_motion_eval.py` from `gpu_ground_segmentation --moving --seed N --obs-csv`. "
            "The sensor drives along the road at 10 m/s (37 scans); a lead car (8-13 m/s) and a following car "
            "(7-12 m/s) drive in its lane, 12 m ahead and behind at the start (dev: 11.5 and 9 m/s). The other "
            "vehicles are parked. Seeds 1-%d are held-out drives. All boxes are completed with the size prior; "
            "a vehicle a tracker has not (yet) tracked gets the single-scan box.\n" % args.seeds)
    section(w, "Dev drive (seed 0)", dev)
    section(w, "Held-out drives (seeds 1-%d)" % args.seeds, held)
    out = os.path.join(ROOT, args.out)
    with open(out + ".csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(dev[0].keys()))
        wr.writeheader()
        wr.writerows(dev + held)
    with open(out + ".md", "w", encoding="utf-8", newline="\n") as f:
        f.write(w.getvalue())
    print(w.getvalue())


if __name__ == "__main__":
    main()
