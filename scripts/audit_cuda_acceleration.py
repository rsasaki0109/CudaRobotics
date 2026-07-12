#!/usr/bin/env python3
"""Rank CUDA sources for likely multi-x optimization opportunities."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def kernel_bodies(text: str) -> list[str]:
    bodies = []
    for match in re.finditer(r"__global__\s+[^;{]+\{", text):
        start = match.end() - 1
        depth = 0
        for index in range(start, len(text)):
            depth += text[index] == "{"
            depth -= text[index] == "}"
            if depth == 0:
                bodies.append(text[start:index + 1])
                break
    return bodies


def inspect(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    kernels = kernel_bodies(text)
    kernel_loops = sum(len(re.findall(r"\bfor\s*\(", body)) for body in kernels)
    nested_loops = sum(len(re.findall(r"for\s*\([^)]*\)\s*\{?[^{}]{0,250}for\s*\(", body, re.S)) for body in kernels)
    atomics = len(re.findall(r"\batomic(?:Add|Min|Max|CAS|Exch)\s*\(", text))
    d2h = len(re.findall(r"cudaMemcpy(?:Async)?\s*\([^;]*cudaMemcpyDeviceToHost", text, re.S))
    h2d = len(re.findall(r"cudaMemcpy(?:Async)?\s*\([^;]*cudaMemcpyHostToDevice", text, re.S))
    syncs = len(re.findall(r"cuda(?:Device|Stream)Synchronize\s*\(", text))
    allocations = len(re.findall(r"cudaMalloc(?:Managed)?\s*\(", text))
    transcendentals = sum(len(re.findall(r"\b(?:sin|cos|atan2|sqrt|exp|log|tanh)f?\s*\(", body)) for body in kernels)
    large_local_arrays = sum(len(re.findall(r"\b(?:float|double|int)\s+\w+\s*\[[A-Z][A-Z0-9_]*\]", body)) for body in kernels)
    host_roundtrip = int(d2h > 0 and h2d > 0)
    score = (3 * kernel_loops + 8 * nested_loops + 5 * host_roundtrip +
             2 * syncs + 2 * atomics + min(allocations, 8) +
             min(transcendentals, 8) + 3 * large_local_arrays)
    signals = []
    if nested_loops: signals.append("nested kernel loops")
    if kernel_loops >= 3: signals.append(f"{kernel_loops} kernel loops")
    if host_roundtrip: signals.append("D2H+H2D round trip")
    if syncs: signals.append(f"{syncs} explicit sync")
    if atomics: signals.append(f"{atomics} atomics")
    if large_local_arrays: signals.append(f"{large_local_arrays} symbolic local arrays")
    if transcendentals >= 4: signals.append(f"{transcendentals} transcendental sites")
    return {
        "file": path.as_posix(), "score": score, "kernels": len(kernels),
        "kernel_loops": kernel_loops, "nested_kernel_loops": nested_loops,
        "d2h": d2h, "h2d": h2d, "syncs": syncs, "atomics": atomics,
        "allocations": allocations, "transcendentals": transcendentals,
        "large_local_arrays": large_local_arrays, "signals": "; ".join(signals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=Path("src"))
    parser.add_argument("--csv", type=Path, default=Path("build/cuda_acceleration_audit.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("build/cuda_acceleration_audit.md"))
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    rows = sorted((inspect(path) for path in args.source.rglob("*.cu")), key=lambda row: (-row["score"], row["file"]))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    lines = ["# CUDA Acceleration Static Audit", "",
             f"Scanned {len(rows)} CUDA translation units. Scores are triage signals, not measured speedup claims.", "",
             "| Rank | File | Score | Kernels | Kernel loops | Nested | Round trips | Syncs | Atomics | Signals |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for rank, row in enumerate(rows[:args.top], 1):
        lines.append(f"| {rank} | `{row['file']}` | {row['score']} | {row['kernels']} | {row['kernel_loops']} | "
                     f"{row['nested_kernel_loops']} | {int(row['d2h'] > 0 and row['h2d'] > 0)} | {row['syncs']} | "
                     f"{row['atomics']} | {row['signals']} |")
    lines += ["", "Interpretation:", "", "- Nested or many per-thread kernel loops suggest algorithmic parallelism is still serial.",
              "- D2H+H2D and explicit synchronization suggest pipeline fusion or device-resident iteration.",
              "- Atomics require contention measurement; their presence alone does not imply a bottleneck.",
              "- Allocation counts are inventory only; source review must confirm whether allocation occurs in a hot loop.",
              "", "## Manually validated remaining opportunities", "",
              "These are source-reviewed hypotheses, not benchmark results. `High` means the code contains a serial or "
              "asymptotically expensive structure that can be replaced; it does not guarantee a particular speedup.", "",
              "| Priority | Algorithms | Source evidence | Optimization direction | Multi-x confidence |",
              "|---|---|---|---|---|",
              "| A | FGR (`gpu_fgr.cu`, `fgr_gpu.cu`) | KNN and feature matching scan every target point/feature, giving dense O(N^2) work | Spatial index or hash; shared-memory tiling/GEMM for descriptors | High at large N |",
              "| A | Constrained MPC (`gpu_constrained_mpc.cu`) | One CUDA thread performs long nested horizon/iLQR loops for a whole problem | Block-per-problem horizon parallelism and batched small-matrix operations | High for long horizons or few problems |",
              "| A | MPC-QP (`gpu_mpc_qp.cu`) | One thread serializes ADMM and forward/back substitution for each agent | Warp/block-per-agent solver or batched triangular solves | High as horizon M grows |",
              "| A | MegaParticles 6DoF (`gpu_megaparticles_6dof.cu`) | Cumulative sum is a one-thread O(N) kernel; likelihood also loops over every scan per particle | CUB DeviceScan plus scan-major/coalesced likelihood evaluation | High for the resampling stage |",
              "| A | Pose graph 3D / bundle adjustment | PCG copies scalar reductions to the host in every iteration; normal equations use many atomics | Device-resident PCG/convergence and block-aggregated assembly | Medium-high for solver-heavy cases |",
              "| A | FilterReg / point-to-plane registration | Dense nested correspondence loops and contended atomic normal-equation updates | Spatial pruning, tiled correspondence, block reductions | Medium-high at large point counts |",
              "| B | Graph/online SLAM family | Repeated synchronization, host round trips, and atomic assembly | Keep iterations device-resident; fuse reductions and aggregate atomics | Medium; profile by graph size |",
              "| B | Diff-MPPI variants and STOMP | Per-thread trajectory loops and synchronization remain, but workload is already broadly parallel | Transposed layouts, fusion, graphs, and selective recomputation | Workload-dependent |",
              "", "## Already demonstrated multi-x acceleration", "",
              "Repository benchmarks already report large GPU gains for several families, so they are not the first targets "
              "for another algorithmic rewrite:", "",
              "| Algorithm | Reported result | Evidence |", "|---|---:|---|",
              "| Batched iLQR | about 140x | `docs/gpu_batched_ilqr.md` |",
              "| KD-tree nearest neighbor | about 175x vs CPU KD-tree; 10,500x vs brute force | `docs/gpu_kdtree_nn.md` |",
              "| SGM stereo | about 46x | `docs/gpu_sgm_stereo.md` |",
              "| Gaussian splatting renderer | 1,381x in the documented comparison | `docs/gaussian_splatting_renderer.md` |",
              "| MPPI control update | up to 6.27x in the controlled benchmark | `docs/results/mppi_control_update_2026-07-12.md` |",
              "", "## Conclusion", "",
              "Not every algorithm has another multi-x gain available: small inputs, memory-bandwidth limits, and already "
              "parallel implementations often cap low-risk tuning near 1.1-2x. The Priority A items are the strongest "
              "remaining multi-x candidates because they expose serial O(N), O(N^2), or host-synchronized iterative work. "
              "Benchmarking each proposed replacement against identical inputs is required before claiming a speedup."]
    args.markdown.parent.mkdir(parents=True, exist_ok=True); args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"scanned {len(rows)} CUDA files; wrote {args.csv} and {args.markdown}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
