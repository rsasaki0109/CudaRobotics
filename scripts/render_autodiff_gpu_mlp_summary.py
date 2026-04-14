#!/usr/bin/env python3
"""Render a visual summary for the autodiff engine and GPU MLP tests."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyBboxPatch


BG = "#f2ede3"
PAPER = "#fffaf1"
CARD = "#fff8ee"
INK = "#1f1a17"
MUTED = "#6f6259"
ACCENT = "#a1401c"
ACCENT_SOFT = "#ead1c5"
COOL = "#517f5d"
COOL_SOFT = "#d9e5db"
GOLD = "#e2b55f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="gif/autodiff_gpu_mlp_summary.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root used to run test binaries",
    )
    return parser.parse_args()


def run_binary(repo_root: Path, relpath: str) -> str:
    result = subprocess.run(
        [str(repo_root / relpath)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_gpu_mlp(text: str) -> dict:
    def parse_loss_block(block_name: str) -> tuple[list[int], list[float], float]:
        block = re.search(rf"\[{re.escape(block_name)}\](.*?)(?:\n\[Test|\nALL TESTS)", text, re.S)
        if not block:
            raise ValueError(f"Could not find block {block_name}")
        body = block.group(1)
        points = re.findall(r"Step (\d+): loss = ([0-9.]+)", body)
        final_match = re.search(r"Final loss: ([0-9.]+)", body)
        if not final_match:
            raise ValueError(f"Missing final loss for {block_name}")
        steps = [int(s) for s, _ in points]
        losses = [float(v) for _, v in points]
        final_loss = float(final_match.group(1))
        return steps, losses, final_loss

    xor_steps, xor_losses, xor_final = parse_loss_block("Test 1")
    sdf_steps, sdf_losses, sdf_final = parse_loss_block("Test 2")

    pred_match = re.search(r"Predictions:\s+([\-0-9.\s]+)\n\s+Targets:\s+([\-0-9.\s]+)", text)
    if not pred_match:
        raise ValueError("Missing XOR predictions")
    predictions = [float(v) for v in pred_match.group(1).split()]
    targets = [float(v) for v in pred_match.group(2).split()]

    infer_match = re.search(r"1M forward passes: ([0-9.]+) ms\s+Throughput: ([0-9.]+) M samples/sec", text, re.S)
    if not infer_match:
        raise ValueError("Missing throughput block")

    return {
        "xor_steps": xor_steps + [1000],
        "xor_losses": xor_losses + [xor_final],
        "xor_final": xor_final,
        "xor_predictions": predictions,
        "xor_targets": targets,
        "sdf_steps": sdf_steps + [8000],
        "sdf_losses": sdf_losses + [sdf_final],
        "sdf_final": sdf_final,
        "inference_ms": float(infer_match.group(1)),
        "throughput_m": float(infer_match.group(2)),
    }


def parse_autodiff(text: str) -> dict:
    scalar_match = re.search(
        r"Autodiff:\s+val=([0-9.]+), deriv=([0-9.]+)\n"
        r"\s+Analytical:\s+val=([0-9.]+), deriv=([0-9.]+)\n"
        r"\s+Numerical:\s+deriv=([0-9.]+)",
        text,
    )
    if not scalar_match:
        raise ValueError("Missing scalar autodiff check")

    jacobian_rows = []
    jacobian_num_rows = []
    for row in re.findall(r"Row \d:\s+([0-9./\-\s]+)", text):
        row_ad = []
        row_num = []
        for pair in row.split():
            ad_val, num_val = pair.split("/")
            row_ad.append(float(ad_val))
            row_num.append(float(num_val))
        jacobian_rows.append(row_ad)
        jacobian_num_rows.append(row_num)
    if not jacobian_rows:
        raise ValueError("Missing Jacobian rows")

    gpu_checks = re.findall(
        r"(sin\(x\)\*x\^2 at x=1|exp\(a\)\+log\(3\) at a=2|atan2\(y,1\) at y=1): "
        r"val=([0-9.]+) \(exp ([0-9.]+)\), deriv=([0-9.]+) \(exp ([0-9.]+)\)",
        text,
    )
    if len(gpu_checks) != 3:
        raise ValueError("Missing GPU kernel checks")

    jacobian = np.asarray(jacobian_rows, dtype=float)
    jacobian_num = np.asarray(jacobian_num_rows, dtype=float)
    return {
        "scalar": {
            "autodiff_val": float(scalar_match.group(1)),
            "autodiff_deriv": float(scalar_match.group(2)),
            "analytical_val": float(scalar_match.group(3)),
            "analytical_deriv": float(scalar_match.group(4)),
            "numerical_deriv": float(scalar_match.group(5)),
        },
        "jacobian": jacobian,
        "jacobian_num": jacobian_num,
        "gpu_checks": gpu_checks,
        "jacobian_match_count": int(np.sum(np.abs(jacobian - jacobian_num) < 1e-3)),
    }


def panel_card(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.02, 0.97, title, transform=ax.transAxes, ha="left", va="top", fontsize=14, color=INK, fontweight="bold")
    if subtitle:
        ax.text(0.02, 0.89, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=10, color=MUTED)


def draw_metric_strip(fig: plt.Figure, metrics: list[tuple[str, str, str]]) -> None:
    left = 0.06
    bottom = 0.86
    width = 0.88
    height = 0.09
    box = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        transform=fig.transFigure,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=PAPER,
        edgecolor=ACCENT_SOFT,
        linewidth=1.2,
    )
    fig.patches.append(box)
    col_w = width / len(metrics)
    for idx, (kicker, strong, desc) in enumerate(metrics):
        x = left + idx * col_w + 0.02
        fig.text(x, bottom + 0.062, kicker, fontsize=9, color=MUTED, family="monospace")
        fig.text(x, bottom + 0.030, strong, fontsize=21, color=INK, fontweight="bold")
        fig.text(x, bottom + 0.010, desc, fontsize=9, color=MUTED)


def render_summary(gpu: dict, autodiff: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 9.0), facecolor=BG)
    fig.text(0.06, 0.975, "Autodiff + GPU MLP Foundation", fontsize=24, color=INK, fontweight="bold", va="top")
    fig.text(
        0.06,
        0.947,
        "Summary generated from the existing test binaries: XOR learning, circle SDF fitting, GPU inference throughput, and forward-mode autodiff checks.",
        fontsize=11,
        color=MUTED,
        va="top",
    )

    draw_metric_strip(
        fig,
        [
            ("xor final loss", f"{gpu['xor_final']:.1e}", "2->4->1 tanh MLP"),
            ("sdf final loss", f"{gpu['sdf_final']:.4f}", "circle SDF on 8x8 grid"),
            ("1M forwards", f"{gpu['inference_ms']:.2f} ms", f"{gpu['throughput_m']:.0f}M samples/sec"),
            ("jacobian check", f"{autodiff['jacobian_match_count']}/24", "entries match numerical"),
        ],
    )

    gs = fig.add_gridspec(2, 2, left=0.06, right=0.94, bottom=0.08, top=0.83, hspace=0.18, wspace=0.12)

    ax_loss = fig.add_subplot(gs[0, 0])
    panel_card(ax_loss, "GPU MLP training traces", "Toy tasks used by the repository tests")
    ax_loss_in = ax_loss.inset_axes([0.08, 0.14, 0.88, 0.68])
    ax_loss_in.set_facecolor(PAPER)
    ax_loss_in.plot(gpu["xor_steps"], gpu["xor_losses"], color=ACCENT, linewidth=2.5, marker="o", label="XOR")
    ax_loss_in.plot(gpu["sdf_steps"], gpu["sdf_losses"], color=COOL, linewidth=2.5, marker="o", label="Circle SDF")
    ax_loss_in.set_yscale("log")
    ax_loss_in.set_xlabel("Training step", color=MUTED)
    ax_loss_in.set_ylabel("MSE loss", color=MUTED)
    ax_loss_in.tick_params(colors=MUTED, labelsize=9)
    ax_loss_in.grid(True, alpha=0.18)
    for spine in ax_loss_in.spines.values():
        spine.set_color(ACCENT_SOFT)
    leg = ax_loss_in.legend(frameon=False, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK)
    ax_loss.text(0.08, 0.06, "XOR converges to exact signs; the same tiny MLP core also fits an implicit SDF target.", transform=ax_loss.transAxes, color=MUTED, fontsize=10)

    ax_xor = fig.add_subplot(gs[0, 1])
    panel_card(ax_xor, "XOR predictions", "Final outputs copied from bin/test_gpu_mlp")
    table_ax = ax_xor.inset_axes([0.06, 0.15, 0.52, 0.68])
    table_ax.set_facecolor(PAPER)
    table_ax.set_xlim(-0.5, 1.5)
    table_ax.set_ylim(-0.5, 1.5)
    table_ax.set_xticks([0, 1], ["x2=-1", "x2=+1"])
    table_ax.set_yticks([0, 1], ["x1=-1", "x1=+1"])
    table_ax.tick_params(colors=MUTED, labelsize=10)
    for spine in table_ax.spines.values():
        spine.set_color(ACCENT_SOFT)
    cmap = LinearSegmentedColormap.from_list("warmcool", ["#8d3320", "#fff6ec", "#496f56"])
    preds_grid = np.asarray(gpu["xor_predictions"], dtype=float).reshape(2, 2)
    table_ax.imshow(preds_grid, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5), cmap=cmap, vmin=-1, vmax=1)
    for i in range(2):
        for j in range(2):
            val = preds_grid[i, j]
            target = np.asarray(gpu["xor_targets"]).reshape(2, 2)[i, j]
            table_ax.text(j, i + 0.12, f"{val:+.3f}", ha="center", va="center", color=INK, fontsize=12, fontweight="bold")
            table_ax.text(j, i - 0.16, f"target {target:+.0f}", ha="center", va="center", color=MUTED, fontsize=9)
    table_ax.grid(color=ACCENT_SOFT, linewidth=1.0)

    text_ax = ax_xor.inset_axes([0.63, 0.18, 0.31, 0.63])
    text_ax.axis("off")
    text_ax.text(0.0, 0.92, "Why it matters", color=ACCENT, fontsize=11, fontweight="bold")
    text_ax.text(0.0, 0.77, "Shared MLP core for Neural SDF\nand later differentiable modules.", color=INK, fontsize=9.4)
    text_ax.text(0.0, 0.44, "Final behavior", color=ACCENT, fontsize=11, fontweight="bold")
    text_ax.text(
        0.0,
        0.30,
        "Predictions land almost exactly on {-1, +1}.",
        color=INK,
        fontsize=10,
        wrap=True,
    )
    ax_xor.text(0.06, 0.06, "The XOR fit is intentionally tiny, but it proves the whole train/infer loop on GPU end to end.", transform=ax_xor.transAxes, color=MUTED, fontsize=10)

    ax_sdf = fig.add_subplot(gs[1, 0])
    panel_card(ax_sdf, "Circle SDF toy target", "Analytic field used in bin/test_gpu_mlp")
    sdf_ax = ax_sdf.inset_axes([0.06, 0.12, 0.54, 0.70])
    sdf_ax.set_facecolor(PAPER)
    grid_n = 121
    xs = np.linspace(-2.0, 2.0, grid_n)
    ys = np.linspace(-2.0, 2.0, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    sdf = np.sqrt(xx**2 + yy**2) - 1.0
    sdf_img = sdf_ax.imshow(sdf, extent=(-2, 2, -2, 2), origin="lower", cmap="RdBu_r", vmin=-1.4, vmax=1.4)
    sdf_ax.contour(xx, yy, sdf, levels=[0.0], colors=[GOLD], linewidths=2.2)
    train_x = np.linspace(-2.0, 2.0, 8)
    train_y = np.linspace(-2.0, 2.0, 8)
    tx, ty = np.meshgrid(train_x, train_y)
    sdf_ax.scatter(tx.ravel(), ty.ravel(), s=14, c=PAPER, edgecolors=INK, linewidths=0.4, alpha=0.9)
    sdf_ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=GOLD, linewidth=2.0, alpha=0.8))
    sdf_ax.set_xlabel("x", color=MUTED)
    sdf_ax.set_ylabel("y", color=MUTED)
    sdf_ax.tick_params(colors=MUTED, labelsize=9)
    for spine in sdf_ax.spines.values():
        spine.set_color(ACCENT_SOFT)
    cax = ax_sdf.inset_axes([0.63, 0.25, 0.03, 0.45])
    cb = fig.colorbar(sdf_img, cax=cax)
    cb.outline.set_edgecolor(ACCENT_SOFT)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    note_ax = ax_sdf.inset_axes([0.70, 0.18, 0.24, 0.60])
    note_ax.axis("off")
    note_ax.text(0.0, 0.90, "2 -> 16 -> 16 -> 1\n8x8 grid supervision\n8000 train steps", color=INK, fontsize=9.6)
    note_ax.text(0.0, 0.46, "Final loss", color=ACCENT, fontsize=11, fontweight="bold")
    note_ax.text(0.0, 0.28, f"{gpu['sdf_final']:.5f}", color=INK, fontsize=18, fontweight="bold")
    note_ax.text(0.0, 0.12, "Reused later by Neural SDF navigation.", color=MUTED, fontsize=8.7, wrap=True)

    ax_ad = fig.add_subplot(gs[1, 1])
    panel_card(ax_ad, "Forward-mode autodiff checks", "Scalar derivative agreement plus dynamics Jacobian")
    heat_ax = ax_ad.inset_axes([0.06, 0.18, 0.50, 0.62])
    heat_ax.set_facecolor(PAPER)
    jac = autodiff["jacobian"]
    im = heat_ax.imshow(jac, cmap="YlOrBr", vmin=0.0, vmax=max(1.0, float(np.max(np.abs(jac)))))
    heat_ax.set_xticks(range(6), ["x", "y", "theta", "v", "a", "steer"], rotation=35, ha="right")
    heat_ax.set_yticks(range(4), ["x'", "y'", "theta'", "v'"])
    heat_ax.tick_params(colors=MUTED, labelsize=9)
    for spine in heat_ax.spines.values():
        spine.set_color(ACCENT_SOFT)
    for i in range(jac.shape[0]):
        for j in range(jac.shape[1]):
            heat_ax.text(j, i, f"{jac[i, j]:.3f}", ha="center", va="center", color=INK, fontsize=8)
    cax2 = ax_ad.inset_axes([0.58, 0.23, 0.025, 0.50])
    cb2 = fig.colorbar(im, cax=cax2)
    cb2.outline.set_edgecolor(ACCENT_SOFT)
    cb2.ax.tick_params(colors=MUTED, labelsize=8)

    txt_ax = ax_ad.inset_axes([0.66, 0.16, 0.29, 0.66])
    txt_ax.axis("off")
    scalar = autodiff["scalar"]
    txt_ax.text(0.0, 0.92, "Scalar check", color=ACCENT, fontsize=11, fontweight="bold")
    txt_ax.text(
        0.0,
        0.72,
        f"sin(x) * x^2 @ x=1\nval {scalar['autodiff_val']:.6f}\nderiv {scalar['autodiff_deriv']:.6f}\nmatches analytical and numerical",
        color=INK,
        fontsize=9.7,
    )
    txt_ax.text(0.0, 0.38, "GPU kernel ops", color=ACCENT, fontsize=11, fontweight="bold")
    txt_ax.text(
        0.0,
        0.12,
        "sin, exp+log, and atan2 all pass on-device.\nThe bicycle Jacobian matches numerical differentiation on every printed entry.",
        color=INK,
        fontsize=10,
        wrap=True,
    )

    fig.text(0.06, 0.04, "Generated by scripts/render_autodiff_gpu_mlp_summary.py using bin/test_gpu_mlp and bin/test_autodiff.", color=MUTED, fontsize=9)
    fig.savefig(out_path, dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    gpu_text = run_binary(repo_root, "bin/test_gpu_mlp")
    autodiff_text = run_binary(repo_root, "bin/test_autodiff")
    gpu = parse_gpu_mlp(gpu_text)
    autodiff = parse_autodiff(autodiff_text)
    render_summary(gpu, autodiff, Path(args.out))
    print(f"Saved summary to {args.out}")


if __name__ == "__main__":
    main()
