"""Clone a PD controller with a Gaussian process, and make it stable with its variance.

The GP learns the map (angles) -> (feedback torques) from a run of the PD controller. It
then replaces the PD controller in the loop, together with the gravity compensation.

The script runs two closed loops:

  1. The pure cloned policy. It diverges. The GP has no speed and no reference, so its
     torque is correct near the training path only.
  2. The same policy plus a term that moves the robot away from uncertainty. It is
     stable. The uncertainty of the GP acts as a map to the training path.

The kernel is periodic with a period of 2*pi, because a joint angle is periodic. The
learned period must come out near 6.283.

Run `python -m gp_learning.tasks.make_robot_datasets` first.

    python -m gp_learning.tasks.clone_torques
    python -m gp_learning.tasks.clone_torques --gif    # also write the animation
"""

import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jax import numpy as jnp

from jax_double_pendulum.dynamics import dynamical_matrices
from jax_double_pendulum.motion_planning import (
    ELLIPSE_PARAMS,
    generate_ellipse_trajectory,
)
from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from robot_control.animation import save_arm_gif
from robot_control.controllers import ctrl_ff_gravity_compensation
from gp_learning.cloning import make_torque_cloning_controller
from gp_learning.data_utils import (
    DATA_DIR,
    OUTPUT_DIR,
    generate_training_data,
    plot_data,
)
from gp_learning.gp_models import periodic_kernel
from gp_learning.wrappers import MultitaskGPRegressor

DATASET = "pd_tracking"
INPUT_COLUMNS = ["th_ts_1", "th_ts_2_rel"]
OUTPUT_COLUMNS = ["tau_fb_ts_1", "tau_fb_ts_2"]

NUM_EPOCHS = 400
SEED = 42
K_VAR = 2.0

SIM_DT = 0.005
SIM_DURATION = 6.0
GRID_SIZE = 100


def train_controller(num_epochs: int):
    df = pd.read_csv(DATA_DIR / f"{DATASET}.csv")
    X, Y = generate_training_data(df, INPUT_COLUMNS, OUTPUT_COLUMNS)
    plot_data(
        df, INPUT_COLUMNS, OUTPUT_COLUMNS, filepath=str(OUTPUT_DIR / "clone_torque_dataset.pdf")
    )

    torch.manual_seed(SEED)
    model = MultitaskGPRegressor(X, Y, kernel_fn=periodic_kernel)
    model.train(num_epochs=num_epochs)
    model.plot_convergence(OUTPUT_DIR / "clone_torque_convergence.pdf")

    period = model.gp.covar_module.base_kernel.period_length
    print(f"Learned period of the kernel: {period.item():.4f} rad (2*pi = 6.2832)")
    return model, X


def plot_policy_surfaces(model: MultitaskGPRegressor):
    """Plot the cloned torque and its uncertainty over the angle plane."""
    angle_range = np.linspace(-np.pi, np.pi, GRID_SIZE)
    th1_grid, th2_grid = np.meshgrid(angle_range, angle_range)
    X_test = torch.tensor(
        np.column_stack((th1_grid.flatten(), th2_grid.flatten())), dtype=torch.float32
    )

    with torch.no_grad():
        prediction = model.predict(X_test)
        mean = prediction.mean.numpy()
        stddev = prediction.stddev.numpy()

    for values, name, label in (
        (mean, "torque", r"$\tau$ [Nm]"),
        (stddev, "uncertainty", r"$\sigma$ [Nm]"),
    ):
        fig = plt.figure(figsize=(11, 4.5))
        for link in range(2):
            ax = fig.add_subplot(1, 2, link + 1, projection="3d")
            ax.plot_surface(
                th1_grid,
                th2_grid,
                values[:, link].reshape(GRID_SIZE, GRID_SIZE),
                cmap="viridis",
            )
            ax.set_xlabel(r"$\theta_{\mathrm{rel},1}$ [rad]")
            ax.set_ylabel(r"$\theta_{\mathrm{rel},2}$ [rad]")
            ax.set_zlabel(f"link {link + 1}, {label}")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"clone_torque_surface_{name}.pdf")
        plt.close(fig)


def plot_uncertainty_map(model: MultitaskGPRegressor, X, filepath):
    """Draw the uncertainty over the angle plane, with the extra torque.

    The training data lies on one closed path, so the variance has one valley. The
    negative gradient of the variance points to that path from every side. This is the
    map that the controller follows.
    """
    angle_range = np.linspace(-np.pi, np.pi, GRID_SIZE)
    th1_grid, th2_grid = np.meshgrid(angle_range, angle_range)
    X_test = torch.tensor(
        np.column_stack((th1_grid.flatten(), th2_grid.flatten())), dtype=torch.float32
    )

    with torch.no_grad():
        stddev = model.predict(X_test).stddev.numpy()
    sigma = stddev.mean(axis=1).reshape(GRID_SIZE, GRID_SIZE)

    gradient_th2, gradient_th1 = np.gradient(sigma**2, angle_range, angle_range)
    step = 7
    u, v = -gradient_th1[::step, ::step], -gradient_th2[::step, ::step]
    length = np.hypot(u, v) + 1e-12

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    field = ax.contourf(th1_grid, th2_grid, sigma, levels=30, cmap="viridis")
    fig.colorbar(field, ax=ax, label=r"$\sigma$ of the cloned torque [Nm]")
    ax.quiver(
        th1_grid[::step, ::step],
        th2_grid[::step, ::step],
        u / length,
        v / length,
        color="white",
        alpha=0.75,
        width=0.004,
    )
    training = np.asarray(X)
    ax.plot(training[:, 0], training[:, 1], "r.", ms=1.5, label="training path")

    ax.set_xlabel(r"$\theta_1$ [rad]")
    ax.set_ylabel(r"$\theta_{\mathrm{rel},2}$ [rad]")
    ax.set_title("The uncertainty of the GP, and the direction of the extra torque")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(filepath, dpi=140)
    plt.close(fig)
    print(f"Wrote {filepath}.")


def run_closed_loop(model: MultitaskGPRegressor, k_var: float, name: str):
    """Simulate the robot with the cloned controller."""
    t_ts = SIM_DT * jnp.arange(int(SIM_DURATION / SIM_DT))
    traj_ts = generate_ellipse_trajectory(rp=ROBOT_PARAMS, t_ts=t_ts, **ELLIPSE_PARAMS)

    ctrl_ff = partial(
        ctrl_ff_gravity_compensation, partial(dynamical_matrices, ROBOT_PARAMS)
    )
    ctrl_fb = make_torque_cloning_controller(model, k_var=k_var)

    sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=traj_ts["th_ts"][0],
        th_d_0=traj_ts["th_d_ts"][0],
        ctrl_ff=ctrl_ff,
        ctrl_fb=ctrl_fb,
        jit_compile=False,
    )

    rmse = report_path_error(traj_ts, sim_ts, name)
    plot_path(traj_ts, sim_ts, name, OUTPUT_DIR / f"clone_torque_path_{name}.pdf")
    return sim_ts, traj_ts, rmse


def report_path_error(traj_ts, sim_ts, name: str) -> float:
    """Print the tip error and give the RMSE.

    The largest error holds the start transient, so the RMSE gives the better picture of
    the tracking.
    """
    error = np.linalg.norm(np.array(sim_ts["x_ts"]) - np.array(traj_ts["x_ts"]), axis=1)
    rmse = float(np.sqrt(np.mean(error**2)))
    print(f"{name}: RMSE {rmse:.3f} m, largest error {error.max():.3f} m")
    return rmse


def plot_path(traj_ts, sim_ts, name, filepath):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(traj_ts["x_ts"][:, 0], traj_ts["x_ts"][:, 1], "k--", label="reference")
    ax.plot(sim_ts["x_ts"][:, 0], sim_ts["x_ts"][:, 1], label="robot")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(name.replace("_", " "))
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--k-var", type=float, default=K_VAR)
    parser.add_argument("--gif", action="store_true", help="also write an animation")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model, X = train_controller(args.epochs)
    plot_policy_surfaces(model)
    plot_uncertainty_map(model, X, OUTPUT_DIR / "clone_torque_uncertainty_map.png")

    pure_ts, traj_ts, pure_rmse = run_closed_loop(
        model, k_var=0.0, name="cloned_policy"
    )
    repel_ts, _, repel_rmse = run_closed_loop(
        model, k_var=args.k_var, name="with_variance_repulsion"
    )

    if args.gif:
        save_arm_gif(
            [
                {
                    "title": f"Cloned policy\nRMSE {pure_rmse:.3f} m",
                    "sim_ts": pure_ts,
                    "traj_ts": traj_ts,
                },
                {
                    "title": f"Cloned policy + uncertainty feedback\n"
                    f"RMSE {repel_rmse:.3f} m",
                    "sim_ts": repel_ts,
                    "traj_ts": traj_ts,
                },
            ],
            OUTPUT_DIR / "clone_torque_comparison.gif",
            step_skip=10,
        )

    print(f"\nFigures are in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
