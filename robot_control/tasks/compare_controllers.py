"""Run the five controllers on the same path and compare them in one figure.

Each controller has its own gains, because each one was tuned alone. The figure shows the
tracking error of each controller and the effort of its feedback term.

    python -m robot_control.tasks.compare_controllers
    python -m robot_control.tasks.compare_controllers --gif
"""

import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from jax_double_pendulum.dynamics import dynamical_matrices
from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from robot_control.animation import save_arm_gif
from robot_control.common import (
    OUTPUT_DIR,
    initial_state,
    make_ellipse_trajectory,
    report_tracking_error,
)
from robot_control.controllers import (
    ctrl_fb_pd,
    ctrl_fb_pd_rel,
    ctrl_ff_feedforward,
    ctrl_ff_gravity_compensation,
    ctrl_ff_pd_plus,
)

CONTROLLERS = (
    ("PD, link angles", ctrl_fb_pd, None, 5000.0, 500.0),
    ("PD, joint angles", ctrl_fb_pd_rel, None, 5000.0, 500.0),
    (
        "PD + gravity compensation",
        ctrl_fb_pd,
        ctrl_ff_gravity_compensation,
        5000.0,
        500.0,
    ),
    ("PD + inverse dynamics", ctrl_fb_pd, ctrl_ff_feedforward, 5000.0, 50.0),
    ("PD+ (Paden-Panja)", ctrl_fb_pd, ctrl_ff_pd_plus, 500.0, 50.0),
)


def run(name, ctrl_fb, ctrl_ff_fn, kp, kd, t_ts, traj_ts):
    th_0, th_d_0 = initial_state(traj_ts)
    ctrl_ff = (
        partial(ctrl_ff_fn, partial(dynamical_matrices, ROBOT_PARAMS))
        if ctrl_ff_fn is not None
        else (lambda th, th_d, th_des, th_d_des, th_dd_des: jnp.zeros((2,)))
    )

    sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=th_0,
        th_d_0=th_d_0,
        th_des_ts=traj_ts["th_ts"],
        th_d_des_ts=traj_ts["th_d_ts"],
        th_dd_des_ts=traj_ts["th_dd_ts"],
        ctrl_ff=ctrl_ff,
        ctrl_fb=partial(ctrl_fb, kp=kp * jnp.eye(2), kd=kd * jnp.eye(2)),
    )
    rmse = report_tracking_error(traj_ts, sim_ts, name)
    effort = float(np.linalg.norm(np.asarray(sim_ts["tau_fb_ts"]), axis=1).mean())
    print(f"Mean norm of the feedback torque: {effort:.1f} Nm")
    return sim_ts, rmse


def plot_comparison(traj_ts, results, filepath):
    """Compare the accuracy against the feedback effort.

    Accuracy alone does not show the interesting result. PD+ is as accurate as the other
    controllers, but its feedback torque is much smaller, because the feedforward term
    does the work.
    """
    time_ts = np.asarray(traj_ts["t_ts"])
    labels = [f"{name}\nkp = {kp:.0f}" for name, _, _, kp, _ in results]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    positions = np.arange(len(results))

    for position, (name, sim_ts, rmse, kp, kd) in zip(positions, results):
        color = colors[position % len(colors)]
        axes[0].barh(position, rmse, color=color, alpha=0.85)
        axes[0].text(rmse + 0.001, position, f"{rmse:.4f} m", va="center", fontsize=9)

        torque = np.linalg.norm(np.asarray(sim_ts["tau_fb_ts"]), axis=1)
        axes[1].plot(
            time_ts,
            torque,
            lw=1.1,
            color=color,
            label=f"{name}, mean {torque.mean():.0f} Nm",
        )

    axes[0].set_yticks(positions, labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.0, 0.085)
    axes[0].set_xlabel("RMSE of the tip [m]")
    axes[0].set_title("Accuracy")
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Norm of the feedback torque [Nm]")
    axes[1].set_title("Effort of the feedback term")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(filepath, dpi=140)
    plt.close(fig)
    print(f"Wrote {filepath}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", action="store_true", help="also write an animation")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_ts, traj_ts = make_ellipse_trajectory()

    results = []
    for name, ctrl_fb, ctrl_ff_fn, kp, kd in CONTROLLERS:
        sim_ts, rmse = run(name, ctrl_fb, ctrl_ff_fn, kp, kd, t_ts, traj_ts)
        results.append((name, sim_ts, rmse, kp, kd))

    plot_comparison(traj_ts, results, OUTPUT_DIR / "controller_comparison.png")

    if args.gif:
        panels = [
            {
                "title": f"{name}\nRMSE {rmse:.4f} m",
                "sim_ts": sim_ts,
                "traj_ts": traj_ts,
            }
            for name, sim_ts, rmse, _, _ in (results[0], results[-1])
        ]
        save_arm_gif(panels, OUTPUT_DIR / "controller_comparison.gif", step_skip=8)


if __name__ == "__main__":
    main()
