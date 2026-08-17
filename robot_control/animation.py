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

# The style of the first arm of a panel and of the second arm, if there is one. The
# second arm is thinner and its trail is broken, so that the first arm stays visible
# below it when the two agree.
ARM_STYLES = (
    {"color": "tab:blue", "alpha": 1.0, "lw": 3.5, "ms": 7},
    {"color": "tab:red", "alpha": 0.7, "lw": 2.0, "ms": 5},
)
TRAIL_STYLES = (
    {"color": "tab:orange", "ls": "-", "lw": 1.8},
    {"color": "tab:red", "ls": "--", "lw": 1.4},
)


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
    """Write a GIF that shows the arm of each panel.

    A panel is a dictionary with `title` and `sim_ts`. It can also hold `traj_ts` for
    the reference path, `sim_hat_ts` for a second arm in the same axes, and `labels`
    for the names of the two arms. The panels get the same axis limits and the same
    time step, so a reader can compare them directly.
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
        labels = panel.get("labels", ("robot", "second model"))
        two_arms = panel.get("sim_hat_ts") is not None
        traj_ts = panel.get("traj_ts")
        reference = None if traj_ts is None else np.asarray(traj_ts["x_ts"])
        target = None

        if reference is not None:
            ax.plot(reference[:, 0], reference[:, 1], "k--", lw=1.0, label="reference")
            (target,) = ax.plot(
                [], [], "o", ms=6, mfc="none", mec="k", label="target now"
            )

        arms = []
        for index, key in enumerate(("sim_ts", "sim_hat_ts")):
            if panel.get(key) is None:
                continue
            elbow, tip = link_positions(panel[key]["th_ts"], rp)
            (trail,) = ax.plot(
                [],
                [],
                alpha=0.9,
                label=None if two_arms else "tip",
                **TRAIL_STYLES[index],
            )
            (arm,) = ax.plot(
                [],
                [],
                marker="o",
                label=labels[index] if two_arms else None,
                **ARM_STYLES[index],
            )
            arms.append((elbow, tip, arm, trail))

        clock = ax.text(0.04, 0.94, "", transform=ax.transAxes, fontsize=9)

        ax.set_xlim(-reach, reach)
        ax.set_ylim(-reach, reach)
        ax.set_aspect("equal")
        ax.set_title(panel["title"], fontsize=11)
        ax.set_xlabel("x [m]")
        ax.grid(True, alpha=0.25)
        updates.append((arms, reference, target, clock))

    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    def draw(step: int):
        start = max(0, step - trail_steps)
        for arms, reference, target, clock in updates:
            for elbow, tip, arm, trail in arms:
                arm.set_data(
                    [0.0, elbow[step, 0], tip[step, 0]],
                    [0.0, elbow[step, 1], tip[step, 1]],
                )
                trail.set_data(tip[start : step + 1, 0], tip[start : step + 1, 1])
            if target is not None:
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
