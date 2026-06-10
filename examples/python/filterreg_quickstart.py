#!/usr/bin/env python3
"""FilterReg quickstart: align a noisy, partially overlapping source cloud."""

import math

import numpy as np

import cudarobotics as cr


def make_lumpy(n, seed=1):
    rng = np.random.default_rng(seed)
    pts = np.empty((n, 3), dtype=np.float32)
    bumps = np.array(
        [
            [0.8, 0.2, 0.5, 0.9, 0.25],
            [-0.3, 0.9, 0.2, 0.7, 0.30],
            [0.1, -0.6, 0.8, 0.8, 0.22],
            [-0.7, -0.4, -0.5, 1.0, 0.28],
            [0.4, 0.3, -0.85, 0.6, 0.20],
        ],
        dtype=np.float32,
    )
    for i in range(n):
        z = rng.uniform(-1.0, 1.0)
        phi = rng.uniform(0.0, 2.0 * math.pi)
        r2 = math.sqrt(max(0.0, 1.0 - z * z))
        dx, dy, dz = r2 * math.cos(phi), r2 * math.sin(phi), z
        radius = (
            2.0
            + 0.35 * math.sin(3.0 * phi) * (1.0 - z * z)
            + 0.30 * dz * dx
            + 0.20 * math.cos(2.0 * phi)
        )
        for bx, by, bz, height, width in bumps:
            ang = 1.0 - (dx * bx + dy * by + dz * bz)
            radius += height * math.exp(-(ang * ang) / (2.0 * width * width))
        pts[i] = radius * np.array([dx, dy, dz], dtype=np.float32)
    return pts


def main():
    target = make_lumpy(8000, seed=1)
    gt_rot = np.array([0.12, -0.18, 0.08], dtype=np.float32)
    gt_trans = np.array([0.35, -0.25, 0.20], dtype=np.float32)
    cx, cy, cz = gt_rot
    c = math.cos
    s = math.sin
    R = np.array(
        [
            [c(cy) * c(cz), -c(cy) * s(cz), s(cy)],
            [s(cx) * s(cy) * c(cz) + c(cx) * s(cz), -s(cx) * s(cy) * s(cz) + c(cx) * c(cz), -s(cx) * c(cy)],
            [-c(cx) * s(cy) * c(cz) + s(cx) * s(cz), c(cx) * s(cy) * s(cz) + s(cx) * c(cz), c(cx) * c(cy)],
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(7)
    keep = rng.uniform(0.0, 1.0, len(target)) <= 0.85
    source = (target @ R.T) + gt_trans + rng.normal(0.0, 0.02, size=target.shape)
    source = source[keep].astype(np.float32)

    registrar = cr.registration.FilterReg()
    rotation, translation, info = registrar.register(target, source)
    print(f"iterations={info['iterations']} rmse={info['final_rmse']:.4f}")
    print("rotation (row-major):", np.array(rotation).reshape(3, 3))
    print("translation:", translation)


if __name__ == "__main__":
    main()
