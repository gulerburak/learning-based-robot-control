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

Run `python -m p3_gp_learning.tasks.make_robot_datasets` first.

    python -m p3_gp_learning.tasks.clone_torques
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
from p2_control.controllers import ctrl_ff_gravity_compensation
from p3_gp_learning.cloning import make_torque_cloning_controller
from p3_gp_learning.data_utils import (
    DATA_DIR,
    OUTPUT_DIR,
    generate_training_data,
    plot_data,
)
from p3_gp_learning.gp_models import periodic_kernel
from p3_gp_learning.wrappers import MultitaskGPRegressor

DATASET = "pd_tracking"
INPUT_COLUMNS = ["th_ts_1", "th_ts_2_rel"]
OUTPUT_COLUMNS = ["tau_fb_ts_1", "tau_fb_ts_2"]

NUM_EPOCHS = 400
SEED = 42
K_VAR = 2.0

SIM_DT = 0.005
SIM_DURATION = 6.0
GRID_SIZE = 100


def train_controller(num_epochs: int) -> MultitaskGPRegressor:
    df = pd.read_csv(DATA_DIR / f"{DATASET}.csv")
    X, Y = generate_training_data(df, INPUT_COLUMNS, OUTPUT_COLUMNS)
    plot_data(
        df, INPUT_COLUMNS, OUTPUT_COLUMNS, filepath=str(OUTPUT_DIR / "3f_dataset.pdf")
    )

    torch.manual_seed(SEED)
    model = MultitaskGPRegressor(X, Y, kernel_fn=periodic_kernel)
    model.train(num_epochs=num_epochs)
    model.plot_convergence(OUTPUT_DIR / "3f_convergence.pdf")

    period = model.gp.covar_module.base_kernel.period_length
    print(f"Learned period of the kernel: {period.item():.4f} rad (2*pi = 6.2832)")
    return model


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
        fig.savefig(OUTPUT_DIR / f"3f_surface_{name}.pdf")
        plt.close(fig)


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

    error = np.linalg.norm(np.array(sim_ts["x_ts"]) - np.array(traj_ts["x_ts"]), axis=1)
    print(f"{name}: largest path error {error.max():.3f} m, last error {error[-1]:.3f} m")

    _plot_path(traj_ts, sim_ts, name, OUTPUT_DIR / f"3f_path_{name}.pdf")
    return sim_ts


def _plot_path(traj_ts, sim_ts, name, filepath):
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
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = train_controller(args.epochs)
    plot_policy_surfaces(model)

    run_closed_loop(model, k_var=0.0, name="cloned_policy")
    run_closed_loop(model, k_var=args.k_var, name="with_variance_repulsion")

    print(f"\nFigures are in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
