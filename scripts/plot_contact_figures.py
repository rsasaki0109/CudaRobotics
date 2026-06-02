#!/usr/bin/env python3
"""Generate the contact-rich Diff-MPPI paper figures.

Produces three publication PDFs into paper/latex/figures/:
  fig_contact_monotone.pdf  success / angular residual vs. number of gradient steps
  fig_cdf_vs_diff.pdf       compute-quality scatter on the smooth 7-DOF task
  fig_box_samples.pdf       box_align success vs. sample budget K (the categorical win)

The numbers are the canonical published values (see paper/cdf_mppi_baseline_results.md
and the tables in paper/latex/diff_mppi.tex), reproduced by:
  bin/benchmark_diff_mppi_pushing_box --k-values 256,1024 --seed-count 8
  bin/benchmark_diff_mppi_pushing_box --scenarios box_align --planners mppi \
      --k-values 2048,4096 --seed-count 8
  bin/benchmark_cdf_mppi_7dof --scenarios 7dof_shelf_reach --seed-count 4

Usage:
  python3 scripts/plot_contact_figures.py [--out-dir paper/latex/figures]
"""

import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D

# === Publication settings (match scripts/generate_paper_figures.py) ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

SINGLE_COL = 3.5  # inches
C_DIFF = "#17becf"   # diff_mppi (teal), matches existing palette
C_MPPI = "#1f77b4"   # vanilla mppi (blue)
C_CDF = "#d62728"    # cdf_mppi (red)
C_FB = "#ff7f0e"     # feedback_mppi_ref (orange)
C_DIFF5 = "#9467bd"  # diff_mppi_5 (purple)


def fig_contact_monotone(out_dir):
    """Two panels: success (box_align) and angular residual (box_pivot) vs grad steps.

    The active ingredient is the gradient step count: holding the sampler fixed
    and adding gradient steps strictly improves the executed pose on both tasks.
    """
    grad_steps = [0, 1, 3, 5]  # 0 == vanilla mppi
    # box_align success at K=1024 (8 seeds)
    align_success = [0.00, 0.00, 0.50, 1.00]
    # box_pivot continuous angular residual at K=1024 (8 seeds); lower is better
    pivot_residual = [0.193, 0.139, 0.124, 0.112]
    pivot_tol = 0.11

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.0, SINGLE_COL * 0.85))

    ax1.plot(grad_steps, align_success, "-o", color=C_DIFF5, markersize=5)
    ax1.set_xlabel("gradient steps $N_g$")
    ax1.set_ylabel("success rate")
    ax1.set_title("(a) box_align: success")
    ax1.set_xticks(grad_steps)
    ax1.set_ylim(-0.05, 1.08)
    ax1.grid(True, alpha=0.3)
    ax1.annotate("vanilla MPPI\n(0 grad steps)", xy=(0, 0.0), xytext=(0.6, 0.30),
                 fontsize=6, color=C_MPPI,
                 arrowprops=dict(arrowstyle="->", color=C_MPPI, lw=0.8))

    ax2.plot(grad_steps, pivot_residual, "-o", color=C_DIFF5, markersize=5)
    ax2.axhline(pivot_tol, color="gray", ls="--", lw=0.8)
    ax2.text(5, pivot_tol + 0.004, "tolerance", ha="right", va="bottom",
             fontsize=6, color="gray")
    ax2.set_xlabel("gradient steps $N_g$")
    ax2.set_ylabel("angular residual [rad]")
    ax2.set_title("(b) box_pivot: orientation error")
    ax2.set_xticks(grad_steps)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # lower (better) is up

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_contact_monotone.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_cdf_vs_diff(out_dir):
    """Compute-quality scatter on 7dof_shelf_reach (the negative-control task).

    CDF-MPPI sits at the top-left (high success, low compute); the Diff-MPPI
    family does not dominate it.
    """
    # (ms/step, success, label, color)
    pts = [
        (0.05, 1.00, "cdf_mppi", C_CDF),
        (0.46, 0.50, "diff_mppi_1", C_DIFF),
        (0.75, 0.25, "diff_mppi_3", C_DIFF5),
        (0.38, 0.25, "mppi", C_MPPI),
        (1.66, 0.00, "feedback_mppi_ref", C_FB),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.9))
    for ms, succ, label, color in pts:
        ax.scatter(ms, succ, s=55, color=color, zorder=3, edgecolor="k", linewidth=0.4)
        dx, dy = 0.04, 0.03
        ha = "left"
        if label == "feedback_mppi_ref":
            dx, ha = -0.04, "right"
        ax.annotate(label, xy=(ms, succ), xytext=(ms + dx, succ + dy),
                    fontsize=6, ha=ha, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("compute [ms/step] (log)")
    ax.set_ylabel("success rate")
    ax.set_title("7dof_shelf_reach (smooth task)")
    ax.set_ylim(-0.08, 1.12)
    ax.grid(True, alpha=0.3, which="both")
    ax.annotate("better", xy=(0.05, 1.00), xytext=(0.13, 0.78), fontsize=6,
                color="gray", arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_cdf_vs_diff.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_box_samples(out_dir):
    """box_align success vs sample budget K: vanilla MPPI cannot buy it.

    diff_mppi_5 succeeds; vanilla MPPI stays at 0 even at 16x the samples.
    """
    ks = [256, 1024, 2048, 4096]
    mppi_succ = [0.00, 0.00, 0.00, 0.00]
    # diff_mppi_5 evaluated at K=256, 1024 (8 seeds); not run at 2048/4096
    diff5_ks = [256, 1024]
    diff5_succ = [0.62, 1.00]

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.9))
    x = range(len(ks))
    ax.plot(list(x), mppi_succ, "-s", color=C_MPPI, markersize=5, label="mppi (sampling only)")
    diff5_x = [ks.index(k) for k in diff5_ks]
    ax.plot(diff5_x, diff5_succ, "-o", color=C_DIFF5, markersize=5,
            label="diff_mppi_5 (5 grad steps)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("sample budget $K$")
    ax.set_ylabel("success rate (box_align)")
    ax.set_title("Samples cannot replace the gradient")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    ax.annotate("$16\\times$ samples,\nstill 0", xy=(3, 0.0), xytext=(2.0, 0.30),
                fontsize=6, color=C_MPPI,
                arrowprops=dict(arrowstyle="->", color=C_MPPI, lw=0.8))
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_box_samples.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_robustness(out_dir):
    """Two panels: box_align success vs contact-model mismatch on two axes.

    (a) contact-mobility (gain) mismatch; (b) object-size (geometry) mismatch.
    diff_mppi_5 @K=1024 vs the strongest sampler mppi @K=4096 (16x samples).
    The shaded band marks where the gradient holds a categorical advantage; the
    mppi curve is flat at 0 across the discriminating range on both axes.
    Numbers are the 8-seed values in paper/cdf_mppi_baseline_results.md.
    """
    # (a) contact-mobility gain scale G
    gain_G = [0.6, 0.7, 0.85, 1.0, 1.2, 1.4, 1.6]
    gain_mppi = [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
    gain_diff = [1.00, 1.00, 1.00, 0.75, 0.50, 0.25, 0.12]
    # (b) object-size scale G
    size_G = [0.7, 0.85, 1.0, 1.15, 1.3]
    size_mppi = [0.00, 0.00, 0.00, 0.00, 0.00]
    size_diff = [0.00, 0.25, 0.75, 0.62, 0.00]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.0, SINGLE_COL * 0.85))

    def panel(ax, G, mppi, diff, band, title, xlabel):
        ax.axvspan(band[0], band[1], color=C_DIFF5, alpha=0.10, lw=0)
        ax.plot(G, diff, "-o", color=C_DIFF5, markersize=5,
                label="diff_mppi_5 (K=1024)")
        ax.plot(G, mppi, "-s", color=C_MPPI, markersize=5,
                label="mppi (K=4096, 16$\\times$)")
        ax.axvline(1.0, color="gray", ls=":", lw=0.8)
        ax.text(1.0, 1.04, "matched", ha="center", va="bottom",
                fontsize=6, color="gray")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("success rate (box_align)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=6)

    panel(ax1, gain_G, gain_mppi, gain_diff, (0.7, 1.4),
          "(a) contact-mobility mismatch", "plant/model gain scale $G$")
    ax1.annotate("task easy\nfor all", xy=(0.6, 1.0), xytext=(0.62, 0.55),
                 fontsize=6, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    panel(ax2, size_G, size_mppi, size_diff, (0.85, 1.15),
          "(b) object-size mismatch", "plant/model box-size scale $G$")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_robustness.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_robustness_pivot(out_dir):
    """box_pivot replication of the two-axis mismatch robustness, on the
    continuous angular residual (the tight ang_tol makes the binary latch
    uninformative). diff_mppi_5 @K=1024 vs mppi @K=4096. The gradient residual
    stays strictly below the sampling floor at every G on both axes; mppi never
    reaches the 0.11 tolerance. Numbers: 8-seed values in
    paper/cdf_mppi_baseline_results.md (box_pivot task-generality subsection).
    """
    tol = 0.11
    gain_G = [0.7, 0.85, 1.0, 1.2, 1.4]
    gain_mppi = [0.260, 0.222, 0.193, 0.163, 0.138]
    gain_diff = [0.171, 0.137, 0.115, 0.106, 0.101]
    size_G = [0.7, 0.85, 1.0, 1.15, 1.3]
    size_mppi = [0.700, 0.248, 0.193, 0.192, 0.196]
    size_diff = [0.700, 0.162, 0.115, 0.111, 0.106]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.0, SINGLE_COL * 0.85))

    def panel(ax, G, mppi, diff, title, xlabel, ymax):
        ax.fill_between(G, diff, mppi, color=C_DIFF5, alpha=0.10, lw=0)
        ax.plot(G, diff, "-o", color=C_DIFF5, markersize=5, label="diff_mppi_5 (K=1024)")
        ax.plot(G, mppi, "-s", color=C_MPPI, markersize=5, label="mppi (K=4096, 16$\\times$)")
        ax.axhline(tol, color="#2ca02c", ls="--", lw=0.9)
        ax.text(G[0], tol + 0.006, "ang tol 0.11", fontsize=6, color="#2ca02c", va="bottom")
        ax.axvline(1.0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("final angular residual [rad]")
        ax.set_title(title)
        ax.set_ylim(0.0, ymax)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=6)

    panel(ax1, gain_G, gain_mppi, gain_diff,
          "(a) contact-mobility mismatch", "plant/model gain scale $G$", 0.30)
    panel(ax2, size_G, size_mppi, size_diff,
          "(b) object-size mismatch", "plant/model box-size scale $G$", 0.75)
    ax2.annotate("both collapse\n(box too small)", xy=(0.7, 0.700), xytext=(0.78, 0.52),
                 fontsize=6, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_robustness_pivot.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def _read_diag_scatter(path):
    cost, netrot = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("cost"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            cost.append(float(parts[0]))
            netrot.append(float(parts[2]))
    return cost, netrot


def fig_mechanism_sampling(out_dir, traj_dir):
    """Why 16x samples cannot rescue vanilla MPPI on the rotation plateau.

    (a) At a stuck box_pivot decision state, each of K=4096 sampled rollouts as
        (net box rotation, cost). Isotropic velocity noise piles up at ~zero
        rotation (the box does not turn unless the push is precisely off-centre);
        the cost-minimizing samples sit in a thin positive-rotation band that is a
        small minority, so the softmax-weighted mean -- dominated by the inactive
        pile -- cannot follow them. The autodiff gradient points straight into the
        low-cost band.
    (b) The fraction of samples that can break the angular-tolerance latch
        (escape_frac) is ~5-9% and ~independent of K from 256 to 4096: the stall is
        structural, not a sample-budget shortfall. Data: --diag-mechanism mode.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.0, SINGLE_COL * 0.85))

    # --- (a) cost vs net rotation scatter at the plateau ---
    spath = os.path.join(traj_dir, "diag_box_pivot_scatter.csv")
    if os.path.exists(spath):
        cost, netrot = _read_diag_scatter(spath)
        ax1.scatter(netrot, cost, s=3, color=C_MPPI, alpha=0.18, lw=0, zorder=2)
        # binned mean-cost curve (the cost-rotation coupling, U-shaped)
        import collections
        binw = 0.05
        agg = collections.defaultdict(list)
        for nr, c in zip(netrot, cost):
            agg[round(math.floor(nr / binw) * binw + binw / 2, 3)].append(c)
        xs = sorted(b for b, v in agg.items() if len(v) >= 8)
        ys = [sum(agg[b]) / len(agg[b]) for b in xs]
        ax1.plot(xs, ys, "-", color="k", lw=1.3, zorder=4, label="mean cost / bin")
        ax1.axvline(0.08, color="#2ca02c", ls="--", lw=0.9, zorder=3)
        ax1.text(0.085, ax1.get_ylim()[1] * 0.92, "rotation needed\nto break latch",
                 fontsize=6, color="#2ca02c", va="top")
        ax1.annotate("64% of samples\nbarely rotate", xy=(0.0, 2.65), xytext=(-0.18, 9.5),
                     fontsize=6, color="gray",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
        ax1.set_xlim(-0.25, 0.35)
        ax1.set_ylim(0, 13)
        ax1.set_xlabel("net box rotation of sample [rad]")
        ax1.set_ylabel("rollout cost")
        ax1.set_title("(a) box_pivot plateau: K=4096 samples")
        ax1.legend(loc="upper left", fontsize=6)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "run --diag-mechanism first", ha="center", transform=ax1.transAxes)

    # --- (b) escape_frac vs K (plateau mean) -- structural, not budget ---
    Ks = [256, 1024, 4096]
    escape = [0.043, 0.089, 0.070]
    xpos = list(range(len(Ks)))
    ax2.bar(xpos, escape, color=C_MPPI, width=0.6, zorder=2)
    for x, e in zip(xpos, escape):
        ax2.text(x, e + 0.004, f"{e:.3f}", ha="center", fontsize=6)
    ax2.axhline(sum(escape) / len(escape), color="gray", ls=":", lw=0.9)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels([f"{k}\n({k//256}$\\times$)" for k in Ks])
    ax2.set_ylim(0, 0.13)
    ax2.set_xlabel("samples $K$ (vanilla MPPI)")
    ax2.set_ylabel("latch-break fraction (plateau)")
    ax2.set_title("(b) starvation is structural in $K$")
    ax2.annotate("16$\\times$ samples\ndoes not help", xy=(2, 0.070), xytext=(0.7, 0.108),
                 fontsize=6, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_mechanism_sampling.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_escape_predictor(out_dir):
    """escape_frac as a PREDICTOR of where the contact gradient helps.

    A matched pair of contact-rotation tasks that both reach sustained contact
    (episode contact_frac 0.26 vs 0.30) but differ in how far past tolerance the
    box must rotate (margin 0.10 vs 0.59 rad):

    (a) In the contact-engaged, still-needs-rotation regime, the fraction of K
        samples that break the angular-tolerance latch (escape_frac) is ~0.62 on
        box_swivel vs ~0.07 on box_pivot -- a 9x gap that is flat across a 16x
        sweep of K (256..4096). Sampling is starved on one task and not the other,
        structurally (not for want of budget).
    (b) The consequence. Final angular error normalized by each task's tolerance
        (1.0 = success threshold): where escape_frac is high (box_swivel), vanilla
        MPPI already lands far inside tolerance and even beats diff_mppi_5 -- the
        gradient is redundant and its perturbation mildly hurts. Where escape_frac
        is low (box_pivot), only the gradient pulls the residual toward tolerance.
        escape_frac predicts the SIGN of the gradient's value. Data: --diag-mechanism
        (panel a) and an 8-seed benchmark, mppi K=4096 vs diff_mppi_5 K=1024 (panel b).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.0, SINGLE_COL * 0.85))
    C_SW = "#2ca02c"   # box_swivel (high escape, sampling suffices) -- green

    # --- (a) engaged escape_frac vs K: high on swivel, low on pivot, both flat ---
    Ks = [256, 1024, 4096]
    sw = [0.6145, 0.6140, 0.6209]   # box_swivel, contact-engaged regime
    pv = [0.0668, 0.0859, 0.0689]   # box_pivot,  contact-engaged regime
    xpos = list(range(len(Ks)))
    w = 0.38
    ax1.bar([x - w / 2 for x in xpos], sw, width=w, color=C_SW, zorder=2,
            label="box_swivel (vanilla wins)")
    ax1.bar([x + w / 2 for x in xpos], pv, width=w, color=C_MPPI, zorder=2,
            label="box_pivot (gradient needed)")
    for x, a, b in zip(xpos, sw, pv):
        ax1.text(x - w / 2, a + 0.012, f"{a:.2f}", ha="center", fontsize=6)
        ax1.text(x + w / 2, b + 0.012, f"{b:.2f}", ha="center", fontsize=6)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([f"{k}\n({k // 256}$\\times$)" for k in Ks])
    ax1.set_ylim(0, 0.78)
    ax1.set_xlabel("samples $K$ (vanilla MPPI)")
    ax1.set_ylabel("latch-break fraction (engaged)")
    ax1.set_title("(a) escape_frac: the predictor")
    ax1.legend(loc="upper center", fontsize=6)
    ax1.grid(True, axis="y", alpha=0.3)

    # --- (b) outcome: final ang error / tolerance (1.0 = success threshold) ---
    tasks = ["box_swivel\n(escape 0.62)", "box_pivot\n(escape 0.07)"]
    mppi_norm = [0.004 / 0.20, 0.193 / 0.11]   # vanilla MPPI K=4096
    diff_norm = [0.076 / 0.20, 0.115 / 0.11]   # diff_mppi_5 K=1024
    xpos2 = list(range(len(tasks)))
    ax2.bar([x - w / 2 for x in xpos2], mppi_norm, width=w, color=C_MPPI, zorder=2,
            label="mppi (K=4096, 16$\\times$)")
    ax2.bar([x + w / 2 for x in xpos2], diff_norm, width=w, color=C_DIFF5, zorder=2,
            label="diff_mppi_5 (K=1024)")
    ax2.axhline(1.0, color="k", ls="--", lw=0.9, zorder=3)
    ax2.text(-0.46, 1.03, "tolerance (success)", fontsize=6, ha="left", va="bottom")
    for x, a, b in zip(xpos2, mppi_norm, diff_norm):
        ax2.text(x - w / 2, a + 0.04, f"{a:.2f}", ha="center", fontsize=6)
        ax2.text(x + w / 2, b + 0.04, f"{b:.2f}", ha="center", fontsize=6)
    ax2.set_xticks(xpos2)
    ax2.set_xticklabels(tasks)
    ax2.set_ylim(0, 2.0)
    ax2.set_ylabel("final ang. error / tolerance")
    ax2.set_title("(b) consequence: where the gradient helps")
    ax2.legend(loc="upper left", fontsize=6)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_escape_predictor.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_sim2sim(out_dir):
    """Sim-to-sim: does the contact-gradient win survive a STRUCTURALLY different
    true plant? The controller's rollout + gradient keep the smooth model; ground
    truth is a hard-contact rigid body (exact non-penetration + Coulomb stick-slip
    friction + box momentum), swept over friction mu. mppi gets K=4096 (16x), the
    gradient gets K=1024.

    (a) Loose tolerance (box_align, ang_tol 0.25): diff_mppi_5 succeeds at every mu
        while vanilla MPPI stays near zero -- the win is NOT an artifact of the
        controller and plant sharing one smooth contact idealization.
    (b) Tight tolerance (box_pivot, ang_tol 0.11): the honest boundary. At low
        friction the gradient lands the residual inside tolerance; as mu grows the
        frictionless controller model becomes too wrong and BOTH methods fail
        (residual -> ~0.44). The gradient cannot compensate for a sufficiently
        wrong contact model. Data: --true-plant hard --mu <mu>, 8 seeds.
    """
    mu = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    align_mppi = [0.00, 0.00, 0.12, 0.12, 0.12, 0.12]   # success
    align_diff = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    pivot_mppi = [0.151, 0.207, 0.403, 0.441, 0.441, 0.440]  # final ang_err
    pivot_diff = [0.102, 0.129, 0.238, 0.402, 0.436, 0.433]
    pivot_tol = 0.11

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.0, SINGLE_COL * 0.85))

    ax1.plot(mu, align_diff, "-o", color=C_DIFF5, markersize=5, label="diff_mppi_5 (K=1024)")
    ax1.plot(mu, align_mppi, "-s", color=C_MPPI, markersize=5, label="mppi (K=4096, 16$\\times$)")
    ax1.fill_between(mu, align_diff, align_mppi, color=C_DIFF5, alpha=0.10, lw=0)
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_xlabel("true-plant Coulomb friction $\\mu$")
    ax1.set_ylabel("success rate")
    ax1.set_title("(a) loose tol (box_align): win survives")
    ax1.legend(loc="center right", fontsize=6)
    ax1.grid(True, alpha=0.3)

    ax2.plot(mu, pivot_diff, "-o", color=C_DIFF5, markersize=5, label="diff_mppi_5 (K=1024)")
    ax2.plot(mu, pivot_mppi, "-s", color=C_MPPI, markersize=5, label="mppi (K=4096, 16$\\times$)")
    ax2.fill_between(mu, pivot_mppi, pivot_diff, color=C_DIFF5, alpha=0.10, lw=0)
    ax2.axhline(pivot_tol, color="k", ls="--", lw=0.9)
    ax2.text(0.02, pivot_tol + 0.012, "tolerance (success)", fontsize=6, va="bottom")
    ax2.axvspan(0.55, 1.0, color="gray", alpha=0.10, lw=0)
    ax2.text(0.78, 0.30, "model too wrong:\nboth fail", fontsize=6, color="gray", ha="center")
    ax2.set_ylim(0, 0.50)
    ax2.set_xlabel("true-plant Coulomb friction $\\mu$")
    ax2.set_ylabel("final angular error [rad]")
    ax2.set_title("(b) tight tol (box_pivot): friction erodes win")
    ax2.legend(loc="lower right", fontsize=6)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_sim2sim.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_fidelity_vs_grad(out_dir):
    """Differentiability vs fidelity: the rebuttal to "just use the right model".

    Against the hard-contact true plant, three arms: vanilla MPPI with the WRONG
    (smooth) model, vanilla MPPI with the EXACT hard model in its rollout (no gradient
    -- the hard solver is non-differentiable), and diff_mppi_5 (smooth model + gradient).
    Both samplers get K=4096 (16x); the gradient gets K=1024.

    (a) Loose tolerance (box_align): both fidelity and the gradient rescue the
        smooth-model sampler; the gradient does it with 4x fewer samples.
    (b) Tight tolerance (box_pivot): the decisive panel. At low model error the
        gradient beats the EXACT-model sampler outright -- directed refinement >
        undirected sampling even with perfect fidelity and 16x the budget. As the
        model error grows the exact model wins on robustness, but note it plateaus
        ABOVE tolerance (~0.13): undirected sampling, even exact, never makes the
        final precise alignment. Only the gradient crosses into tolerance, and only
        where the smooth model is accurate enough. Data: --true-plant hard, 8 seeds.
    """
    mu = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    C_HARD = "#ff7f0e"   # exact-hard-model sampler (orange)
    # (a) box_align success
    al_mppi = [0.00, 0.00, 0.12, 0.12, 0.12, 0.12]
    al_hard = [1.00, 1.00, 0.88, 1.00, 1.00, 1.00]
    al_diff = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    # (b) box_pivot final ang_err
    pv_mppi = [0.151, 0.207, 0.403, 0.441, 0.441, 0.440]
    pv_hard = [0.143, 0.199, 0.147, 0.135, 0.129, 0.137]
    pv_diff = [0.102, 0.129, 0.238, 0.402, 0.436, 0.433]
    pv_tol = 0.11

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.0, SINGLE_COL * 0.85))

    ax1.plot(mu, al_diff, "-o", color=C_DIFF5, markersize=5, label="diff_mppi_5: smooth+grad (K=1024)")
    ax1.plot(mu, al_hard, "-^", color=C_HARD, markersize=5, label="mppi: EXACT model (K=4096)")
    ax1.plot(mu, al_mppi, "-s", color=C_MPPI, markersize=5, label="mppi: smooth model (K=4096)")
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_xlabel("true-plant Coulomb friction $\\mu$")
    ax1.set_ylabel("success rate")
    ax1.set_title("(a) loose tol (box_align)")
    ax1.legend(loc="center right", fontsize=5.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(mu, pv_diff, "-o", color=C_DIFF5, markersize=5, label="diff_mppi_5: smooth+grad (K=1024)")
    ax2.plot(mu, pv_hard, "-^", color=C_HARD, markersize=5, label="mppi: EXACT model (K=4096)")
    ax2.plot(mu, pv_mppi, "-s", color=C_MPPI, markersize=5, label="mppi: smooth model (K=4096)")
    ax2.axhline(pv_tol, color="k", ls="--", lw=0.9)
    ax2.text(0.98, pv_tol + 0.008, "tolerance", fontsize=6, va="bottom", ha="right")
    ax2.annotate("grad beats EXACT model\n(differentiability > fidelity)",
                 xy=(0.1, 0.115), xytext=(0.20, 0.30), fontsize=5.5, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax2.annotate("exact model plateaus\nabove tolerance", xy=(0.8, 0.129), xytext=(0.42, 0.055),
                 fontsize=5.5, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax2.set_ylim(0, 0.48)
    ax2.set_xlabel("true-plant Coulomb friction $\\mu$")
    ax2.set_ylabel("final angular error [rad]")
    ax2.set_title("(b) tight tol (box_pivot)")
    ax2.legend(loc="upper left", fontsize=5.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_fidelity_vs_grad.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def _read_traj(path):
    """Read a --dump-traj CSV; return (meta dict, list of (px,py,ox,oy,oth))."""
    meta = {}
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                for tok in line[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta[k] = v
                continue
            if line.startswith("px"):
                continue
            parts = line.strip().split(",")
            if len(parts) == 5:
                rows.append(tuple(float(x) for x in parts))
    return meta, rows


def _draw_box(ax, ox, oy, oth, hx, hy, color, alpha, lw=1.0, ls="-"):
    rect = mpatches.Rectangle((-hx, -hy), 2 * hx, 2 * hy, fill=False,
                              edgecolor=color, alpha=alpha, lw=lw, ls=ls)
    t = Affine2D().rotate(oth).translate(ox, oy) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    # a short heading tick so orientation is legible
    hxw = (hx + 0.04)
    ax.plot([ox, ox + hxw * math.cos(oth)], [oy, oy + hxw * math.sin(oth)],
            color=color, alpha=alpha, lw=lw)


def fig_contact_filmstrip(out_dir, traj_dir):
    """Real box-pose filmstrip: vanilla MPPI (stuck) vs diff_mppi_5 (aligned).

    Reads trajectories dumped by:
      bin/benchmark_diff_mppi_pushing_box --dump-traj <traj_dir>/box_traj
    """
    specs = [("box_traj_mppi.csv", "mppi (sampling only)", C_MPPI),
             ("box_traj_diff_mppi_5.csv", "diff_mppi_5", C_DIFF5)]
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 2.0, SINGLE_COL * 0.95))
    for ax, (fname, title, color) in zip(axes, specs):
        meta, rows = _read_traj(os.path.join(traj_dir, fname))
        hx, hy = float(meta.get("hx", 0.35)), float(meta.get("hy", 0.18))
        gx, gy, gth = (float(x) for x in meta["goal"].split(","))
        n = len(rows)
        # overlay ~6 snapshots, fading from light (start) to solid (end)
        idxs = sorted(set([round(i * (n - 1) / 5) for i in range(6)]))
        for j, i in enumerate(idxs):
            px, py, ox, oy, oth = rows[i]
            a = 0.28 + 0.72 * (j / (len(idxs) - 1))
            _draw_box(ax, ox, oy, oth, hx, hy, color, a, lw=1.1)
        # pusher path
        ax.plot([r[0] for r in rows], [r[1] for r in rows], color="gray",
                lw=0.7, alpha=0.7, zorder=1)
        # goal pose (dashed green)
        _draw_box(ax, gx, gy, gth, hx, hy, "#2ca02c", 0.9, lw=1.2, ls="--")
        ax.plot(gx, gy, "*", color="#2ca02c", ms=8, zorder=5)
        ok = meta.get("success", "0") == "1"
        ax.set_title("%s -- %s (%s steps)" %
                     (title, "reached" if ok else "stuck", meta.get("steps", "?")),
                     fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlim(0.9, 2.7)
        ax.set_ylim(0.7, 2.9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("box_align: target pose (green dashed) vs executed box poses",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_contact_filmstrip.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="paper/latex/figures")
    ap.add_argument("--traj-dir", default="build")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.traj_dir, "box_traj_mppi.csv")):
        fig_contact_filmstrip(args.out_dir, args.traj_dir)
    else:
        print("skip filmstrip: run --dump-traj first")
    fig_contact_monotone(args.out_dir)
    fig_cdf_vs_diff(args.out_dir)
    fig_box_samples(args.out_dir)
    fig_robustness(args.out_dir)
    fig_robustness_pivot(args.out_dir)
    fig_mechanism_sampling(args.out_dir, args.traj_dir)
    fig_escape_predictor(args.out_dir)
    fig_sim2sim(args.out_dir)
    fig_fidelity_vs_grad(args.out_dir)


if __name__ == "__main__":
    main()
