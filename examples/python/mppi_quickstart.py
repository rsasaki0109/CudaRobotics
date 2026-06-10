#!/usr/bin/env python3
import sys

import numpy as np

import cudarobotics as cr


SIZE_X = 200
SIZE_Y = 200
RESOLUTION = 0.05


def make_costmap():
    costmap = np.zeros((SIZE_Y, SIZE_X), dtype=np.uint8)
    wx0, wx1 = int(4.9 / RESOLUTION), int(5.1 / RESOLUTION)
    gy0, gy1 = int(4.0 / RESOLUTION), int(6.0 / RESOLUTION)
    costmap[:gy0, wx0:wx1] = 254
    costmap[gy1:, wx0:wx1] = 254
    return costmap


def render_gif(costmap, trajectory, output):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import PillowWriter
    except ImportError:
        print("matplotlib/pillow not installed; skipping GIF render")
        return False

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(costmap, origin="lower", extent=(0, 10, 0, 10), cmap="gray_r")
    ax.plot([1, 9], [5, 5], "C0--", linewidth=1)
    line, = ax.plot([], [], "C3", linewidth=2)
    point, = ax.plot([], [], "o", color="C3")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    writer = PillowWriter(fps=20)
    with writer.saving(fig, output, dpi=120):
        for i in range(1, len(trajectory) + 1):
            xy = np.asarray(trajectory[:i])
            line.set_data(xy[:, 0], xy[:, 1])
            point.set_data([xy[-1, 0]], [xy[-1, 1]])
            writer.grab_frame()
    plt.close(fig)
    return True


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cudarobotics_mppi_quickstart.gif"
    costmap = make_costmap()
    path_x = np.arange(1.0, 9.05, 0.1, dtype=np.float32)
    path = np.stack([path_x, np.full_like(path_x, 5.0)], axis=1).astype(np.float32)
    planner = cr.MppiPlanner(batch_size=2048, time_steps=56, model_dt=0.05)

    state = np.array([1.0, 5.0, 0.0], dtype=np.float32)
    trajectory = [state[:2].copy()]
    reached = False
    for _ in range(500):
        v, vy, w, info = planner.compute(
            state,
            costmap,
            path,
            (9.0, 5.0, 0.0),
            resolution=RESOLUTION,
            goal_is_final=True,
        )
        if info["all_colliding"] and not info["retreating"]:
            raise RuntimeError("all sampled trajectories collided before a retreat was available")
        yaw = state[2]
        state[0] += 0.05 * (v * np.cos(yaw) - vy * np.sin(yaw))
        state[1] += 0.05 * (v * np.sin(yaw) + vy * np.cos(yaw))
        state[2] = np.arctan2(np.sin(yaw + 0.05 * w), np.cos(yaw + 0.05 * w))
        trajectory.append(state[:2].copy())
        if np.linalg.norm(state[:2] - np.array([9.0, 5.0], dtype=np.float32)) < 0.25:
            reached = True
            break

    if not reached:
        raise RuntimeError(f"goal not reached; final_state={state.tolist()}")
    print(f"final_state={state.tolist()} steps={len(trajectory) - 1}")
    if render_gif(costmap, trajectory, output):
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
