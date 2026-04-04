#!/usr/bin/env python3
"""Analyze gradient freshness: how much new information does the gradient provide?

Compares the MPPI update direction (sampled → final without gradient) vs the
gradient direction on static (corner_turn) and dynamic (dynamic_slalom) tasks.

If the gradient provides "fresh" information on dynamic tasks (low alignment with
MPPI update), that explains why the hybrid controller helps specifically there.
"""

import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


def load_trace(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def main():
    trace_path = "build/benchmark_diff_mppi_freshness_trace.csv"
    out_dir = "build/plots_paper"
    os.makedirs(out_dir, exist_ok=True)

    rows = load_trace(trace_path)

    scenarios = {}
    for r in rows:
        sc = r["scenario"]
        if sc not in scenarios:
            scenarios[sc] = []
        scenarios[sc].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    colors = {"corner_turn": "#2171b5", "dynamic_slalom": "#e6550d"}
    labels = {"corner_turn": "Static (corner_turn)", "dynamic_slalom": "Dynamic (dynamic_slalom)"}

    # ---- Panel A: Gradient magnitude by episode step ----
    ax = axes[0]
    for sc_name, sc_rows in scenarios.items():
        # Group by episode_step, take horizon_step=0 (first action)
        step_data = {}
        for r in sc_rows:
            es = int(r["episode_step"])
            hs = int(r["horizon_step"])
            if hs == 0:
                step_data[es] = float(r["grad_norm"])

        steps = sorted(step_data.keys())
        norms = [step_data[s] for s in steps]
        ax.plot(steps, norms, color=colors.get(sc_name, "gray"),
                label=labels.get(sc_name, sc_name), alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Episode Step", fontsize=10)
    ax.set_ylabel("Gradient Norm (first action)", fontsize=10)
    ax.set_title("(a) Gradient magnitude over time", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # ---- Panel B: Correction magnitude (delta_norm) by episode step ----
    ax = axes[1]
    for sc_name, sc_rows in scenarios.items():
        step_data = {}
        for r in sc_rows:
            es = int(r["episode_step"])
            hs = int(r["horizon_step"])
            if hs == 0:
                step_data[es] = float(r["delta_norm"])

        steps = sorted(step_data.keys())
        deltas = [step_data[s] for s in steps]
        ax.plot(steps, deltas, color=colors.get(sc_name, "gray"),
                label=labels.get(sc_name, sc_name), alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Episode Step", fontsize=10)
    ax.set_ylabel("Correction Norm (first action)", fontsize=10)
    ax.set_title("(b) Autodiff correction over time", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # ---- Panel C: Cosine similarity between MPPI update and gradient ----
    ax = axes[2]
    for sc_name, sc_rows in scenarios.items():
        step_data = {}
        for r in sc_rows:
            es = int(r["episode_step"])
            hs = int(r["horizon_step"])
            if hs == 0:
                # MPPI update direction: (sampled_accel, sampled_steer) - previous
                # Gradient direction: (grad_accel, grad_steer)
                # Delta direction: (delta_accel, delta_steer) = final - sampled
                da = float(r["delta_accel"])
                ds = float(r["delta_steer"])
                ga = float(r["grad_accel"])
                gs = float(r["grad_steer"])

                # Cosine similarity between delta (correction applied) and gradient direction
                delta_norm = math.sqrt(da*da + ds*ds + 1e-12)
                grad_norm = math.sqrt(ga*ga + gs*gs + 1e-12)
                cos_sim = (da*ga + ds*gs) / (delta_norm * grad_norm + 1e-12)
                step_data[es] = cos_sim

        steps = sorted(step_data.keys())
        sims = [step_data[s] for s in steps]
        # Smooth with moving average
        window = 5
        smoothed = []
        for i in range(len(sims)):
            lo = max(0, i - window // 2)
            hi = min(len(sims), i + window // 2 + 1)
            smoothed.append(sum(sims[lo:hi]) / (hi - lo))
        ax.plot(steps, smoothed, color=colors.get(sc_name, "gray"),
                label=labels.get(sc_name, sc_name), alpha=0.8, linewidth=1.5)

    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Episode Step", fontsize=10)
    ax.set_ylabel("Cosine Similarity (correction vs gradient)", fontsize=10)
    ax.set_title("(c) Gradient alignment with correction", fontsize=11, fontweight="bold")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"gradient_freshness.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close()


if __name__ == "__main__":
    main()
