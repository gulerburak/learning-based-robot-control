"""Training, evaluation and figures for the two networks."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from p1_vision_state_estimation.models import CNNTheta, CNNTrig, count_parameters

LOSS_FN = nn.MSELoss(reduction="mean")

LEARNING_RATES = {"theta": 1e-3, "trig": 1e-2}
MODELS = {"theta": CNNTheta, "trig": CNNTrig}


def manual_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _predict_angle(outputs: torch.Tensor, model_type: str) -> torch.Tensor:
    """Get the angle in rad from the raw output of a network."""
    if model_type == "theta":
        return outputs.squeeze(-1)
    return torch.atan2(outputs[:, 0], outputs[:, 1])


def angular_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Give the absolute angle error, with the wrap at the full circle.

    The labels of the dataset are in [0, 2*pi), and `atan2` gives a value in [-pi, pi].
    A direct difference of the two therefore counts a full circle as an error. The
    difference goes through sine and cosine, so that 0.1 rad and 2*pi + 0.1 rad are the
    same angle.
    """
    difference = predicted - target
    return torch.abs(torch.atan2(torch.sin(difference), torch.cos(difference)))


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, model_type: str
) -> Tuple[float, float]:
    """Give the mean loss and the mean angle error over a loader.

    The loss uses the output space of the model, so the two models have different loss
    units. The error is always the mean absolute angle error in rad.
    """
    model.eval()
    loss_sum, error_sum = 0.0, 0.0

    for x, theta, trig in loader:
        outputs = model(x)
        target = theta if model_type == "theta" else trig
        loss_sum += LOSS_FN(outputs, target).item()

        angle_hat = _predict_angle(outputs, model_type)
        error_sum += angular_error(angle_hat, theta.squeeze(-1)).mean().item()

    return loss_sum / len(loader), error_sum / len(loader)


def train_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    seed: int,
    learning_rate: float = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train one network and give the network and its history."""
    manual_seed(seed)
    model = MODELS[model_type]()
    learning_rate = learning_rate or LEARNING_RATES[model_type]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "val_error": []}

    for epoch in (pbar := tqdm(range(num_epochs), desc=f"{model_type} seed {seed}")):
        model.train()
        train_loss = 0.0
        for x, theta, trig in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            target = theta if model_type == "theta" else trig
            loss = LOSS_FN(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_error = evaluate_model(model, val_loader, model_type)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_error"].append(val_error)
        pbar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_error:.4f} rad")

    return model, history


def run_experiment(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    num_runs: int,
    checkpoint_dir: Path,
) -> Tuple[np.ndarray, List[Dict[str, List[float]]], nn.Module]:
    """Train the same network `num_runs` times, with a different seed in each run."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    test_errors = np.zeros(num_runs)
    histories = []
    model = None

    print(f"\n=== {model_type} model ===")
    print(f"Trainable parameters: {count_parameters(MODELS[model_type]())}")

    for run in range(num_runs):
        model, history = train_model(
            model_type, train_loader, val_loader, num_epochs, seed=run
        )
        test_loss, test_error = evaluate_model(model, test_loader, model_type)
        test_errors[run] = test_error
        histories.append(history)

        print(
            f"Run {run}: test loss {test_loss:.4}, mean test error {test_error:.4} rad."
        )
        torch.save(model.state_dict(), checkpoint_dir / f"{model_type}_run-{run}.pth")

    print(
        f"Test error of the {model_type} model across {num_runs} runs: "
        f"{test_errors.mean():.4} +- {test_errors.std():.4} rad."
    )
    return test_errors, histories, model


def plot_loss_curves(
    histories: Dict[str, List[Dict[str, List[float]]]], filepath: Path
):
    """Plot the training loss and the validation error of every run."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for model_type, runs in histories.items():
        for run, history in enumerate(runs):
            label = f"{model_type}, run {run}"
            axes[0].semilogy(history["train_loss"], label=label)
            axes[1].plot(history["val_error"], label=label)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation error [rad]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


@torch.no_grad()
def plot_error_against_angle(
    models: Dict[str, nn.Module], test_loader: DataLoader, filepath: Path
):
    """Plot the absolute angle error against the true angle.

    The direct model has its largest error at the two ends of the range, because the
    angle wraps there.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for model_type, model in models.items():
        model.eval()
        angles, errors = [], []
        for x, theta, trig in test_loader:
            angle = theta.squeeze(-1)
            angle_hat = _predict_angle(model(x), model_type)
            angles.append(angle.numpy())
            errors.append(angular_error(angle_hat, angle).numpy())
        ax.scatter(
            np.concatenate(angles), np.concatenate(errors), s=4, alpha=0.4, label=model_type
        )

    ax.set_xlabel(r"True angle $\theta$ [rad]")
    ax.set_ylabel("Absolute error [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
