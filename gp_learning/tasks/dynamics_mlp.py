"""Learn the same forward dynamics with an MLP, and compare it against the GP.

The MLP fits the data as well as the GP where data exists. The difference is outside the
data. The GP falls back to its zero prior and reports a large standard deviation. The MLP
extrapolates with its last linear pieces, gives a confident answer, and that answer can
disagree with the physics.

Run `python -m gp_learning.tasks.make_robot_datasets` first.

    python -m gp_learning.tasks.dynamics_mlp
"""

import argparse

import pandas as pd
import torch

from gp_learning.data_utils import DATA_DIR, OUTPUT_DIR, generate_training_data
from gp_learning.mlp_model import MLPRegressor
from gp_learning.tasks.dynamics_gp import (
    GRID_SIZE,
    INPUT_COLUMNS,
    OUTPUT_COLUMNS,
    make_phase_grid,
    plot_phase_portrait,
)

NUM_EPOCHS = 200
SEED = 42


def run_dataset(name: str, num_epochs: int, grid):
    theta_grid, theta_dot_grid, X_test = grid

    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    X, Y = generate_training_data(df, INPUT_COLUMNS, OUTPUT_COLUMNS)

    torch.manual_seed(SEED)
    model = MLPRegressor(X, Y)
    model.train(num_epochs=num_epochs)
    model.plot_convergence(OUTPUT_DIR / f"mlp_dynamics_convergence_{name}.pdf")

    print(f"\n=== {name} ===")
    print(f"Final RMSE: {model.rmse_loss_vec[-1]:.4f} rad/s^2")

    mean_grid = model.predict(X_test)[:, 0].view(GRID_SIZE, GRID_SIZE).numpy()
    plot_phase_portrait(
        theta_grid,
        theta_dot_grid,
        mean_grid,
        None,
        f"MLP phase portrait, {name.replace('_', ' ')} data",
        OUTPUT_DIR / f"mlp_dynamics_phase_portrait_{name}.pdf",
    )


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
