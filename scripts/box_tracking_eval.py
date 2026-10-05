#!/usr/bin/env python3
"""Multi-frame box tracking of gpu_ground_segmentation (--sequence), dev vs held-out.

The sensor drives along the road (37 scans, 1 m apart). Each car / van cluster
is tracked in the world frame; a track accumulates the voxels of its clusters
and refits its L-shape box to the voxels seen in at least K scans (--trk-hits).
This script runs the dev scene (seed 0) and held-out scenes (seeds 1..N) for
several K, and compares the tracked boxes with the single-scan ones on the
observations of the cars and the van, per observation and per seed (exact sign
tests), and counts identity switches.

Writes <out>.csv (all observations, K = 3) and <out>.md (the report).
"""

import argparse
import csv
import io
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from box_fitting_heldout import ROOT, find_binary, sign_test  # noqa: E402

VEHICLES = {"0", "1", "2", "4"}
METHODS = [
    ("lshape", "single-scan L-shape"),
    ("prior_mlp", "single-scan L-shape + size prior (learned class)"),
    ("trk_lshape", "tracked L-shape"),
    ("trk_prior", "tracked L-shape + size prior"),
]
COMPARISONS = [
    ("trk_lshape", "lshape", "iou", False),
    ("trk_prior", "prior_mlp", "iou", False),
    ("trk_prior", "prior_mlp", "centre", True),
]
LABEL = dict(METHODS)
METRIC = {"iou": "BEV IoU", "centre": "centre error"}


def run(binary, seed, hits, tmp_csv):
    cmd = [binary, "--no-video", "--sequence", "--seed", str(seed), "--trk-hits", str(hits), "--obs-csv", tmp_csv]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(tmp_csv, newline="") as f:
        return [r for r in csv.DictReader(f) if r["box"] in VEHICLES]


def switches(rows):
    n = 0
    last = {}
    for r in rows:
        key = (r["seed"], r["box"])
        t = int(r["track"])
        if t < 0:
            continue
        if key in last and last[key] != t:
            n += 1
        last[key] = t
    return n


def summary(rows):
    n = len(rows)
    out = []
    for key, label in METHODS:
        iou = [float(r["iou_" + key]) for r in rows]
        out.append((label, sum(iou) / n, sum(v >= 0.5 for v in iou),
                    sum(float(r["centre_" + key]) for r in rows) / n,
                    sum(float(r["yaw_" + key]) for r in rows) / n))
    return out


def compare(rows):
    out = []
    seeds = sorted({r["seed"] for r in rows})
    for a, b, metric, lower in COMPARISONS:
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
        for sd in seeds:
            d = sum(float(r["%s_%s" % (metric, a)]) - float(r["%s_%s" % (metric, b)]) for r in rows if r["seed"] == sd)
            if abs(d) < 1e-9:
                continue
            if (d < 0) == lower:
                sw += 1
            else:
                sl += 1
        diff = sum(float(r["%s_%s" % (metric, a)]) - float(r["%s_%s" % (metric, b)]) for r in rows) / len(rows)
        out.append((LABEL[a], LABEL[b], METRIC[metric], diff, w, l, t, sign_test(w, l), sw, sl, sign_test(sw, sl)))
    return out


def section(w, name, rows):
    w.write("\n## %s (%d vehicle observations, %d identity switches)\n\n" % (name, len(rows), switches(rows)))
    w.write("| Box | BEV IoU | IoU >= 0.5 | centre error | heading error |\n|---|---:|---:|---:|---:|\n")
    for label, iou, good, ctr, yaw in summary(rows):
        w.write("| %s | %.3f | %d | %.2f m | %.2f deg |\n" % (label, iou, good, ctr, yaw))
    multi = len({r["seed"] for r in rows}) > 1
    w.write("\n| Comparison | metric | mean difference | observations better / worse / tie | sign test p |")
    w.write(" seeds better / worse | seed sign test p |\n" if multi else "\n")
    w.write("|---|---|---:|---:|---:|" + ("---:|---:|\n" if multi else "\n"))
    for a, b, m, diff, wn, ls, t, p, sw, sl, sp in compare(rows):
        w.write("| %s vs %s | %s | %+.3f | %d / %d / %d | %.2g |" % (a, b, m, diff, wn, ls, t, p))
        w.write(" %d / %d | %.2g |\n" % (sw, sl, sp) if multi else "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=20, help="held-out seeds 1..N")
    ap.add_argument("--hits", default="1,3,5", help="--trk-hits values; 3 is the default reported in full")
    ap.add_argument("--out", default="docs/results/box_tracking_2026-10-05")
    args = ap.parse_args()
    binary = find_binary()
    tmp = os.path.join(ROOT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    tmp_csv = os.path.join(tmp, "box_trk_obs.csv")
    hits = [int(h) for h in args.hits.split(",")]
    runs = {}
    for k in hits:
        dev = run(binary, 0, k, tmp_csv)
        held = []
        for seed in range(1, args.seeds + 1):
            held += run(binary, seed, k, tmp_csv)
        runs[k] = (dev, held)
        print("K = %d done" % k)
    w = io.StringIO()
    w.write("# LiDAR box tracking along a drive: dev scene vs held-out scenes\n\n")
    w.write("Generated by `scripts/box_tracking_eval.py` from `gpu_ground_segmentation --sequence --seed N "
            "--trk-hits K --obs-csv`. The sensor drives along y = 0.5 m from x = -12 to 24 m (37 scans, 1 m apart). "
            "Seed 0 is the dev scene; seeds 1-%d are held-out scenes. Only the observations of the cars and the van "
            "count. Before a track has history, its box is the single-scan box.\n" % args.seeds)
    w.write("\n## Voxel threshold K (a voxel counts once seen in K scans)\n\n")
    w.write("| K | dev tracked + prior IoU | held-out tracked + prior IoU | held-out centre error | "
            "held-out identity switches |\n|---:|---:|---:|---:|---:|\n")
    for k in hits:
        dev, held = runs[k]
        sd, sh = summary(dev), summary(held)
        w.write("| %d | %.3f | %.3f | %.2f m | %d |\n" % (k, sd[3][1], sh[3][1], sh[3][3], switches(held)))
    w.write("\nSingle-scan L-shape + size prior (learned class), for reference: dev %.3f, held-out %.3f IoU.\n"
            % (summary(runs[hits[0]][0])[1][1], summary(runs[hits[0]][1])[1][1]))
    main_k = 3 if 3 in runs else hits[0]
    dev, held = runs[main_k]
    section(w, "Dev scene, K = %d" % main_k, dev)
    section(w, "Held-out scenes (seeds 1-%d), K = %d" % (args.seeds, main_k), held)
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
