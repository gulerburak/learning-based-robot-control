"""Learn the forward dynamics of the robot with a multi-output sparse GP.

The model maps (speeds, angles, torques) to the angular accelerations. The result is
shown as a phase portrait: the flow lines give the motion, and the colour gives the
standard deviation of the model.

The two datasets show the main property of a Gaussian process. With the large-oscillation
data the model is sure everywhere. With the small-oscillation data the model is sure near
the training area only. Away from it the mean falls back to the zero prior, so the
predicted acceleration is zero and the flow lines become horizontal. The colour makes
that visible before it can cause a problem.

Run `python -m p3_gp_learning.tasks.make_robot_datasets` first.

    python -m p3_gp_learning.tasks.dynamics_gp
"""

import argparse

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from p3_gp_learning.data_utils import (
    DATA_DIR,
    OUTPUT_DIR,
    generate_training_data,
    plot_data,
)
from p3_gp_learning.wrappers import MultitaskGPRegressor

INPUT_COLUMNS = ["th_d_ts_1", "th_d_ts_2", "th_ts_1", "th_ts_2", "tau_ts_1", "tau_ts_2"]
OUTPUT_COLUMNS = ["th_dd_ts_1", "th_dd_ts_2"]

NUM_EPOCHS = 30
SEED = 42
GRID_SIZE = 100
SPEED_LIMIT = 8.0


def make_phase_grid(grid_size: int = GRID_SIZE, speed_limit: float = SPEED_LIMIT):
    """Make a grid of states with zero torque.

    The second link follows the first one, so the plot shows two dimensions only.
    """
    theta_range = np.linspace(-np.pi, np.pi, grid_size)
    theta_dot_range = np.linspace(-speed_limit, speed_limit, grid_size)
    theta_grid, theta_dot_grid = np.meshgrid(theta_range, theta_dot_range)

    zeros = np.zeros_like(theta_grid)
    input_points = np.column_stack(
        (
            theta_dot_grid.flatten(),
            theta_dot_grid.flatten(),
            theta_grid.flatten(),
            theta_grid.flatten(),
            zeros.flatten(),
            zeros.flatten(),
        )
    )
    return theta_grid, theta_dot_grid, torch.tensor(input_points, dtype=torch.float32)


def plot_phase_portrait(
    theta_grid, theta_dot_grid, mean_grid, std_grid, title, filepath
):
    fig, ax = plt.subplots(figsize=(7, 5))

    if std_grid is not None:
        mesh = ax.pcolormesh(theta_grid, theta_dot_grid, std_grid, cmap="viridis")
        fig.colorbar(mesh, ax=ax, label=r"Standard deviation [rad/s$^2$]")

    ax.streamplot(
        theta_grid, theta_dot_grid, theta_dot_grid, mean_grid, density=2, color="black"
    )
    ax.axvline(-np.pi / 2, color="red", linestyle="--", alpha=0.6)
    ax.axvline(np.pi / 2, color="red", linestyle="--", alpha=0.6)
    ax.axhline(0.0, color="red", linestyle="--", alpha=0.6)

    ax.set_xlabel(r"$\theta_1$ [rad]")
    ax.set_ylabel(r"$\dot{\theta}_1$ [rad/s]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def run_dataset(name: str, num_epochs: int, grid):
    theta_grid, theta_dot_grid, X_test = grid

    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    X, Y = generate_training_data(df, INPUT_COLUMNS, OUTPUT_COLUMNS)
    plot_data(
        df,
        INPUT_COLUMNS,
        OUTPUT_COLUMNS,
        filepath=str(OUTPUT_DIR / f"3d_dataset_{name}.pdf"),
    )

    torch.manual_seed(SEED)
    model = MultitaskGPRegressor(X, Y)
    model.train(num_epochs=num_epochs)

    print(f"\n=== {name} ===")
    model.print_info()
    model.plot_convergence(OUTPUT_DIR / f"3d_convergence_{name}.pdf")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model.predict(X_test)
        mean_grid = predictions.mean[:, 0].view(GRID_SIZE, GRID_SIZE).numpy()
        std_grid = predictions.stddev[:, 0].view(GRID_SIZE, GRID_SIZE).numpy()

    plot_phase_portrait(
        theta_grid,
        theta_dot_grid,
        mean_grid,
        std_grid,
        f"GP phase portrait, {name.replace('_', ' ')} data",
        OUTPUT_DIR / f"3d_phase_portrait_{name}.pdf",
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid = make_phase_grid()

    for name in ("big_oscillation", "small_oscillation"):
        run_dataset(name, args.epochs, grid)

    print(f"\nFigures are in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
