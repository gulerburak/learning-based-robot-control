"""A small MLP for the same regression problem as the GP.

The MLP gives a point prediction only. It says nothing about its own uncertainty. That is
the main difference against the Gaussian process.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

HIDDEN_SIZE = 1000
LEARNING_RATE = 0.01
BATCH_SIZE = 256


class MLP(nn.Module):
    """One hidden layer with a ReLU activation."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressor:
    """Train the MLP with the mean squared error."""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, batch_size: int = BATCH_SIZE):
        self.X = X.double()
        self.Y = Y.double()
        self.batch_size = batch_size
        self.rmse_loss_vec = np.array([])
        self.model = MLP(X.shape[1], Y.shape[1]).double()

    def train(self, num_epochs: int = 200, learning_rate: float = LEARNING_RATE):
        loader = DataLoader(
            TensorDataset(self.X, self.Y), batch_size=self.batch_size, shuffle=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        self.rmse_loss_vec = np.zeros(num_epochs)
        self.model.train()
        for epoch in (pbar := tqdm(range(num_epochs), desc="train MLP")):
            loss_sum = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()

            self.rmse_loss_vec[epoch] = np.sqrt(loss_sum / len(loader))
            pbar.set_postfix(rmse=f"{self.rmse_loss_vec[epoch]:.4f}")

        self.model.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x.double())

    def plot_convergence(self, filepath: Path):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(self.rmse_loss_vec)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"RMSE [rad/s$^2$]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filepath)
        plt.close(fig)
