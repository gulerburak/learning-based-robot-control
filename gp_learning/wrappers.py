"""A trainer around the multi-output sparse GP.

The class holds the model, the likelihood and the training loop. It also gives the
variance and the gradient of the variance, which the cloned controllers need.
"""

from pathlib import Path
from typing import Callable

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from gp_learning.gp_models import MultitaskGPModel, matern_kernel

LEARNING_RATE = 0.01
NUM_INDUCING = 100
BATCH_SIZE = 256


class MultitaskGPRegressor:
    """Train a multi-output sparse GP on a table of inputs and labels.

    Args:
        X: shape (num_samples, num_features)
        Y: shape (num_samples, num_tasks)
        kernel_fn: the kernel factory from `gp_models`
        n_ind: the number of inducing points. They start at random training points.
    """

    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        kernel_fn: Callable = matern_kernel,
        n_ind: int = NUM_INDUCING,
        batch_size: int = BATCH_SIZE,
    ):
        self.X = X.double()
        self.Y = Y.double()
        self.num_features = X.shape[1]
        self.num_tasks = Y.shape[1]
        self.batch_size = batch_size
        self.elbo_loss_vec = np.array([])

        self.inducing_points = X[torch.randperm(len(X))[:n_ind]]
        self.gp = MultitaskGPModel(
            self.num_tasks, self.inducing_points, kernel_fn=kernel_fn
        ).double()
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.num_tasks
        ).double()

    def train(self, num_epochs: int = 30, learning_rate: float = LEARNING_RATE):
        """Maximise the evidence lower bound (ELBO) with Adam."""
        loader = DataLoader(
            TensorDataset(self.X, self.Y), batch_size=self.batch_size, shuffle=True
        )
        self.gp.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [{"params": self.gp.parameters()}, {"params": self.likelihood.parameters()}],
            lr=learning_rate,
        )
        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.gp, num_data=self.Y.size(0)
        )

        self.elbo_loss_vec = np.zeros(num_epochs)
        for epoch in (pbar := tqdm(range(num_epochs), desc="train GP")):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = -self.mll(self.gp(x_batch), y_batch)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.elbo_loss_vec[epoch] = -self.mll(self.gp(self.X), self.Y).item()
            pbar.set_postfix(loss=f"{self.elbo_loss_vec[epoch]:.4f}")

        self.gp.eval()
        self.likelihood.eval()

    def predict(self, x: torch.Tensor):
        """Give the predictive distribution, with the noise of the likelihood."""
        return self.likelihood(self.gp(x.double()))

    def variance(self, x: torch.Tensor) -> torch.Tensor:
        """Give the variance of the model, without the noise of the likelihood."""
        return self.gp(x).variance

    def variance_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Give d(variance_i) / d(input_j).

        Returns:
            shape (num_tasks, num_features)
        """
        jac = jacobian(self.variance, x.double()).detach()
        return jac.reshape(self.num_tasks, self.num_features)

    def print_info(self):
        with torch.no_grad():
            print("Lengthscale:\n", self.gp.covar_module.base_kernel.lengthscale)
            print("Output scale:\n", torch.sqrt(self.gp.covar_module.outputscale))
            print("ELBO:\n", self.mll(self.gp(self.X), self.Y).item())

    def plot_convergence(self, filepath: Path):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.elbo_loss_vec)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Negative ELBO")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)
