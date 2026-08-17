"""Animations of the arm and of the learning process, written as GIF files.

The GIF format keeps the figures usable in a document or in a README. `PillowWriter`
makes them, so no other program is necessary.
"""

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from jax_double_pendulum.robot_parameters import ROBOT_PARAMS

FPS = 20
DPI = 100
TRAIL_STEPS = 400


def link_positions(th_ts, rp: Dict = ROBOT_PARAMS):
    """Give the elbow position and the tip position for a series of link angles."""
    th = np.asarray(th_ts)
    elbow = rp["l1"] * np.stack([np.cos(th[:, 0]), np.sin(th[:, 0])], axis=-1)
    tip = elbow + rp["l2"] * np.stack([np.cos(th[:, 1]), np.sin(th[:, 1])], axis=-1)
    return elbow, tip


def save_arm_gif(
    panels: Sequence[Dict],
    filepath: Path,
    rp: Dict = ROBOT_PARAMS,
    step_skip: int = 10,
    trail_steps: int = TRAIL_STEPS,
    fps: int = FPS,
    dpi: int = DPI,
    panel_size: float = 4.0,
    limit: float = None,
):
    """Write a GIF that shows one arm for each panel.

    A panel is a dictionary with `title`, `sim_ts` and `traj_ts`. The panels get the
    same axis limits and the same time step, so a reader can compare them directly.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    reach = limit if limit is not None else rp["l1"] + rp["l2"]
    num_steps = len(panels[0]["sim_ts"]["th_ts"])
    frames = list(range(0, num_steps, step_skip))
    time_ts = np.asarray(panels[0]["sim_ts"]["t_ts"])

    fig, axes = plt.subplots(
        1, len(panels), figsize=(panel_size * len(panels), panel_size + 0.4), dpi=dpi
    )
    axes = np.atleast_1d(axes)
    updates: List = []

    for ax, panel in zip(axes, panels):
        elbow, tip = link_positions(panel["sim_ts"]["th_ts"], rp)
        reference = np.asarray(panel["traj_ts"]["x_ts"])

        ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.0, label="reference")
        (target,) = ax.plot([], [], "o", ms=6, mfc="none", mec="k", label="target now")
        (trail,) = ax.plot([], [], "-", lw=1.6, color="tab:orange", label="tip")
        (arm,) = ax.plot([], [], "o-", lw=3.5, ms=7, color="tab:blue")
        clock = ax.text(0.04, 0.94, "", transform=ax.transAxes, fontsize=9)

        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("x [m]")
        ax.grid(True, alpha=0.25)
        updates.append((elbow, tip, reference, target, trail, arm, clock))

    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    def draw(step: int):
        for elbow, tip, reference, target, trail, arm, clock in updates:
            arm.set_data(
                [0.0, elbow[step, 0], tip[step, 0]],
                [0.0, elbow[step, 1], tip[step, 1]],
            )
            start = max(0, step - trail_steps)
            trail.set_data(tip[start : step + 1, 0], tip[start : step + 1, 1])
            target.set_data([reference[step, 0]], [reference[step, 1]])
            clock.set_text(f"t = {time_ts[step]:.2f} s")

    animation = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    animation.save(filepath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wrote {filepath}.")


def save_ilc_gif(
    traj_ts: Dict,
    ilc_its: Dict,
    filepath: Path,
    num_frames: int = 60,
    fps: int = 10,
    dpi: int = DPI,
):
    """Write a GIF of the ILC learning: the path of one run against the iteration.

    The error falls quickly in the first runs, so the iterations of the frames are
    spaced on a logarithmic scale.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    reference = np.asarray(traj_ts["x_ts"])
    paths = np.asarray(ilc_its["x_its"])
    error = np.linalg.norm(paths - reference[None], axis=-1)
    rmse = np.sqrt(np.mean(error**2, axis=1))

    num_its = paths.shape[0]
    steps = np.unique(
        np.round(np.geomspace(1, num_its, num=num_frames)).astype(int) - 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=dpi)

    axes[0].plot(reference[:, 0], reference[:, 1], "k--", lw=1.2, label="reference")
    axes[0].plot(
        paths[0, :, 0], paths[0, :, 1], lw=1.2, color="0.75", label="first iteration"
    )
    (path_line,) = axes[0].plot([], [], lw=1.6, color="tab:blue", label="robot")
    axes[0].set_xlim(reference[:, 0].min() - 0.6, reference[:, 0].max() + 0.6)
    axes[0].set_ylim(reference[:, 1].min() - 0.6, reference[:, 1].max() + 0.6)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)
    # The placeholder holds the room for the title, so `tight_layout` keeps it visible.
    title = axes[0].set_title("Iteration 0000, RMSE 0.0000 m")

    axes[1].semilogy(np.arange(1, num_its + 1), rmse, color="tab:blue")
    (marker,) = axes[1].plot([], [], "o", color="tab:red")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("RMSE of the tip [m]")
    axes[1].grid(True, alpha=0.25, which="both")
    fig.tight_layout()

    def draw(it: int):
        path_line.set_data(paths[it, :, 0], paths[it, :, 1])
        marker.set_data([it + 1], [rmse[it]])
        title.set_text(f"Iteration {it + 1}, RMSE {rmse[it]:.4f} m")

    animation = FuncAnimation(fig, draw, frames=steps, interval=1000 / fps)
    animation.save(filepath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wrote {filepath}.")
