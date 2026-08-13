"""Fit a tensile test of steel with an exact GP and with a sparse GP.

The data comes from a tensile test machine. It gives the load against the position of the
clamp. The curve has an elastic part, a yield point, and then plastic flow.

The exact GP uses all 373 samples. The sparse GP uses 10 inducing points only. The
comparison shows the cost of the approximation: the sparse model cannot follow the sharp
yield point, so it gets a larger lengthscale and more noise.

    python -m gp_learning.tasks.gp_tensile
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

from gp_learning.data_utils import OUTPUT_DIR
from gp_learning.gp_models import ExactGPModel, SVGPModel

DATA_FILE = Path("data") / "tensile_strength.txt"
NUM_INDUCING = 10
EXACT_EPOCHS = 500
EXACT_LR = 0.1
SVGP_EPOCHS = 10000
SVGP_LR = 0.01
SEED = 42


def load_data():
    """Read the file of the test machine.

    The file has a junk first line, so the header is on line 1. The last row is the break
    of the sample, and it is not part of the curve.
    """
    steel = pd.read_csv(DATA_FILE, sep="\t", header=1)
    x = steel["Position(mm)"].to_numpy()[:-1]
    y = steel["Load(kN)"].to_numpy()[:-1]
    return torch.tensor(x), torch.tensor(y)


def train_exact_gp(train_x, train_y, num_epochs: int):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=EXACT_LR)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in (pbar := tqdm(range(num_epochs), desc="exact GP")):
        optimizer.zero_grad()
        # The marginal log likelihood is maximised, so its negative is minimised.
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            lengthscale=f"{model.covar_module.base_kernel.lengthscale.item():.3f}",
            noise=f"{likelihood.noise.item():.4f}",
        )

    return model, likelihood


def train_svgp(train_x, train_y, num_epochs: int, num_inducing: int):
    inducing_points = torch.linspace(
        torch.min(train_x), torch.max(train_x), num_inducing
    ).double()
    model = SVGPModel(inducing_points).double()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=SVGP_LR
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=1024, shuffle=True)
    for _ in (pbar := tqdm(range(num_epochs), desc="sparse GP")):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = -mll(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return model, likelihood


def plot_fit(train_x, train_y, model, likelihood, title, filepath, inducing=None):
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(torch.min(train_x), torch.max(train_x), 100).double()
        observed_pred = likelihood(model(test_x))
        lower, upper = observed_pred.confidence_region()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_x.numpy(), train_y.numpy(), "k.", markersize=3, label="measurement")
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b", label="mean")
    ax.fill_between(
        test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.3, label=r"$\pm 2\sigma$"
    )
    if inducing is not None:
        ax.plot(inducing, np.zeros_like(inducing), "kx", label="inducing points")

    ax.set_xlabel("Position [mm]")
    ax.set_ylabel("Load [kN]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EXACT_EPOCHS)
    parser.add_argument("--svgp-epochs", type=int, default=SVGP_EPOCHS)
    parser.add_argument("--num-inducing", type=int, default=NUM_INDUCING)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    train_x, train_y = load_data()
    print(f"{len(train_x)} samples.")

    model, likelihood = train_exact_gp(train_x, train_y, args.epochs)
    print("\n=== exact GP ===")
    print("Horizontal lengthscale:", model.covar_module.base_kernel.lengthscale.item())
    print("Vertical lengthscale:", model.covar_module.raw_outputscale.item())
    print("Likelihood noise:", likelihood.noise.item())
    plot_fit(
        train_x, train_y, model, likelihood, "Exact GP", OUTPUT_DIR / "exact_gp.pdf"
    )

    torch.manual_seed(SEED)
    svgp_model, svgp_likelihood = train_svgp(
        train_x, train_y, args.svgp_epochs, args.num_inducing
    )
    inducing = (
        svgp_model.variational_strategy.inducing_points.detach().squeeze().numpy()
    )
    print(f"\n=== sparse GP, {args.num_inducing} inducing points ===")
    print(
        "Horizontal lengthscale:",
        svgp_model.covar_module.base_kernel.lengthscale.item(),
    )
    print("Vertical lengthscale:", svgp_model.covar_module.raw_outputscale.item())
    print("Likelihood noise:", svgp_likelihood.noise.item())
    plot_fit(
        train_x,
        train_y,
        svgp_model,
        svgp_likelihood,
        f"Sparse GP, {args.num_inducing} inducing points",
        OUTPUT_DIR / "sparse_gp.pdf",
        inducing=inducing,
    )
    print(f"\nFigures are in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
