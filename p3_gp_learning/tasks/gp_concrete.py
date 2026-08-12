"""Predict the compressive strength of concrete with a sparse GP that has 8 inputs.

The task compares a constant mean against a zero mean. The comparison is about safety,
not only about accuracy:

  * The constant mean gives a smaller error, because the strength of concrete is not near
    zero.
  * The zero mean is safer. Far from the data it predicts a small strength, so it warns.
    The constant mean predicts a large strength there, and its lower confidence bound can
    stay above the true strength. That is an unsafe prediction.

The script prints the fraction of unsafe predictions for both models.

    python -m p3_gp_learning.tasks.gp_concrete
"""

import argparse
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from p3_gp_learning.data_utils import OUTPUT_DIR
from p3_gp_learning.gp_models import SVGPModel

DATA_FILE = Path("data") / "concrete_data.csv"
NUM_INDUCING = 300
NUM_EPOCHS = 2000
LEARNING_RATE = 0.01
BATCH_SIZE = 256
TEST_RATIO = 0.2
SPLIT_SEED = 42
MODEL_SEED = 43


def load_data():
    """Read the table and split it into a training set and a test set."""
    df = pd.read_csv(DATA_FILE)
    data = df.to_numpy()
    x, y = data[:, :-1], data[:, -1]

    np.random.seed(SPLIT_SEED)
    num_test = int(TEST_RATIO * len(x))
    test_idx = np.random.choice(len(x), num_test, replace=False)
    train_idx = np.setdiff1d(np.arange(len(x)), test_idx)

    print(f"{len(x)} samples: {len(train_idx)} train, {len(test_idx)} test.")
    print("Inputs:", list(df.columns[:-1]))
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def train_model(train_x, train_y, constant_mean: bool, num_epochs: int):
    torch.manual_seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)

    inducing_points = train_x[torch.randperm(len(train_x))[:NUM_INDUCING]]
    model = SVGPModel(inducing_points, constant_mean=constant_mean, ard=True).double()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}],
        lr=LEARNING_RATE,
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True
    )
    name = "constant mean" if constant_mean else "zero mean"
    for _ in (pbar := tqdm(range(num_epochs), desc=name)):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = -mll(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    model.eval()
    likelihood.eval()
    return model, likelihood


def evaluate_model(model, likelihood, train_x, train_y, test_x, test_y):
    """Print the error, the calibration and the fraction of unsafe predictions."""
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        train_pred = likelihood(model(train_x))
        test_pred = likelihood(model(test_x))

    results = {}
    for name, pred, target in (
        ("train", train_pred, train_y),
        ("test", test_pred, test_y),
    ):
        std = pred.stddev
        error = torch.abs(pred.mean - target)

        results[f"mae_{name}"] = error.mean().item()
        # The true value must be inside the 2*sigma interval in about 95 % of the cases.
        results[f"calibration_{name}"] = (error < 2 * std).float().mean().item()
        # An unsafe prediction says that the concrete is stronger than it is.
        results[f"unsafe_{name}"] = (
            ((pred.mean - 2 * std) > target).float().mean().item()
        )

    print(f"  MAE:          train {results['mae_train']:.4f}, test {results['mae_test']:.4f} MPa")
    print(f"  Calibration:  train {results['calibration_train']:.4f}, test {results['calibration_test']:.4f}")
    print(f"  Unsafe share: train {results['unsafe_train']:.4f}, test {results['unsafe_test']:.4f}")
    return train_pred, test_pred, results


def plot_predictions(test_pred, test_y, filepath, num_shown=60):
    """Plot the prediction and its 2*sigma interval against the true value."""
    index = np.arange(num_shown)
    mean = test_pred.mean.numpy()[:num_shown]
    std = test_pred.stddev.numpy()[:num_shown]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(index, mean, yerr=2 * std, fmt="*", label=r"prediction $\pm 2\sigma$")
    ax.plot(index, test_y.numpy()[:num_shown], "r.", label="true value")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Compressive strength [MPa]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_x, train_y, test_x, test_y = load_data()

    tensors = [torch.from_numpy(a).double() for a in (train_x, train_y, test_x, test_y)]
    train_x, train_y, test_x, test_y = tensors

    for constant_mean in (True, False):
        name = "constant mean" if constant_mean else "zero mean"
        model, likelihood = train_model(train_x, train_y, constant_mean, args.epochs)

        print(f"\n=== {name} ===")
        _, test_pred, _ = evaluate_model(
            model, likelihood, train_x, train_y, test_x, test_y
        )
        print("  ARD lengthscales:", model.covar_module.base_kernel.lengthscale.detach().numpy())
        print("  Output scale:", model.covar_module.outputscale.item())
        if constant_mean:
            print("  Constant mean:", model.mean_module.constant.item())

        suffix = "constant_mean" if constant_mean else "zero_mean"
        plot_predictions(test_pred, test_y, OUTPUT_DIR / f"3b_predictions_{suffix}.pdf")

    print(f"\nFigures are in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
