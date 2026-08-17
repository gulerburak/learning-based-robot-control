"""Training, evaluation and figures for the two networks."""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vision_state_estimation.models import CNNTheta, CNNTrig, count_parameters

LOSS_FN = nn.MSELoss(reduction="mean")

LEARNING_RATES = {"theta": 1e-3, "trig": 1e-2}
MODELS = {"theta": CNNTheta, "trig": CNNTrig}
MODEL_LABELS = {"theta": "direct angle", "trig": "sine and cosine"}
MODEL_COLORS = {"theta": "tab:blue", "trig": "tab:orange"}
TRUE_COLOR = "limegreen"


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


def _draw_needle(ax, angle: float, size: int, color: str, label: str = None, lw=2.2):
    """Draw a line from the pivot in the direction of the angle.

    The angle is zero when the link points up and it increases counterclockwise. The row
    index of an image grows downwards, so both components get a minus sign.
    """
    centre = (size - 1) / 2
    length = 0.45 * size
    (line,) = ax.plot(
        [centre, centre - length * np.sin(angle)],
        [centre, centre - length * np.cos(angle)],
        color=color,
        lw=lw,
        label=label,
    )
    return line


def _samples_over_the_circle(test_loader: DataLoader, count: int) -> np.ndarray:
    """Give indices of the test set, spread over the full circle."""
    subset = test_loader.dataset
    indices = np.asarray(subset.indices)
    angles = np.asarray([subset.dataset.angle(index) for index in indices])
    order = np.argsort(angles)
    picks = np.round(np.linspace(0, len(order) - 1, count)).astype(int)
    return indices[order[picks]]


@torch.no_grad()
def _predict_from_image(
    models: Dict[str, nn.Module], x: torch.Tensor
) -> Dict[str, float]:
    batch = x.unsqueeze(0)
    return {
        model_type: float(_predict_angle(model(batch), model_type)[0])
        for model_type, model in models.items()
    }


def _absolute_error(predicted: float, target: float) -> float:
    return float(angular_error(torch.tensor(predicted), torch.tensor(target)))


@torch.no_grad()
def plot_predictions(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    filepath: Path,
    count: int = 8,
):
    """Show the input of the network and the angle that each model reads from it."""
    dataset = test_loader.dataset.dataset
    indices = _samples_over_the_circle(test_loader, count)

    fig, axes = plt.subplots(1, count, figsize=(1.7 * count, 2.9))
    for ax, index in zip(axes, indices):
        x, theta, _ = dataset[index]
        angle = float(theta)
        predicted = _predict_from_image(models, x)
        size = x.shape[-1]

        ax.imshow(x[0].numpy(), cmap="gray")
        _draw_needle(ax, angle, size, TRUE_COLOR, label="true")
        for model_type, angle_hat in predicted.items():
            _draw_needle(
                ax,
                angle_hat,
                size,
                MODEL_COLORS[model_type],
                label=MODEL_LABELS[model_type],
                lw=1.6,
            )

        errors = "\n".join(
            f"{MODEL_LABELS[m]}: {_absolute_error(a, angle):.2f} rad"
            for m, a in predicted.items()
        )
        ax.set_title(f"true {angle:.2f} rad", fontsize=9)
        ax.set_xlabel(errors, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.7)
    fig.suptitle("The 24x24 input, the true angle, and the angle that each model reads")
    fig.tight_layout()
    fig.savefig(filepath, dpi=140)
    plt.close(fig)
    print(f"Wrote {filepath}.")


@torch.no_grad()
def save_prediction_gif(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    filepath: Path,
    count: int = 90,
    fps: int = 12,
):
    """Write a GIF that turns the link over the full circle.

    The needle of the direct model leaves the link where the angle wraps. The needle of
    the sine-cosine model stays on it.
    """
    dataset = test_loader.dataset.dataset
    samples = []
    for index in _samples_over_the_circle(test_loader, count):
        x, theta, _ = dataset[index]
        samples.append((x, float(theta), _predict_from_image(models, x)))

    model_types = list(models.keys())
    size = samples[0][0].shape[-1]
    centre, length = (size - 1) / 2, 0.45 * size

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), dpi=80)
    picture = axes[0].imshow(samples[0][0][0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    needles = {"true": _draw_needle(axes[0], 0.0, size, TRUE_COLOR, label="true")}
    for model_type in model_types:
        needles[model_type] = _draw_needle(
            axes[0],
            0.0,
            size,
            MODEL_COLORS[model_type],
            label=MODEL_LABELS[model_type],
            lw=1.8,
        )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.7)
    # The placeholder holds the room for the title, so `tight_layout` keeps it visible.
    title = axes[0].set_title("True angle 0.00 rad")

    positions = np.arange(len(model_types))
    bars = axes[1].barh(
        positions,
        np.zeros(len(model_types)),
        color=[MODEL_COLORS[m] for m in model_types],
    )
    texts = [axes[1].text(0.0, p, "", va="center", fontsize=9) for p in positions]
    axes[1].set_yticks(positions, [MODEL_LABELS[m] for m in model_types])
    axes[1].set_xlim(0.0, np.pi)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Absolute angle error [rad]")
    axes[1].set_title("Error of this image")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.tight_layout()

    def draw(frame: int):
        x, angle, predicted = samples[frame]
        picture.set_data(x[0].numpy())
        title.set_text(f"True angle {angle:.2f} rad")

        for name, value in [("true", angle)] + [(m, predicted[m]) for m in model_types]:
            needles[name].set_data(
                [centre, centre - length * np.sin(value)],
                [centre, centre - length * np.cos(value)],
            )

        for bar, text, model_type in zip(bars, texts, model_types):
            error = _absolute_error(predicted[model_type], angle)
            bar.set_width(error)
            text.set_position((error + 0.06, text.get_position()[1]))
            text.set_text(f"{error:.2f} rad")

    animation = FuncAnimation(fig, draw, frames=len(samples), interval=1000 / fps)
    animation.save(filepath, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wrote {filepath}.")


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
            np.concatenate(angles),
            np.concatenate(errors),
            s=4,
            alpha=0.4,
            color=MODEL_COLORS[model_type],
            label=MODEL_LABELS[model_type],
        )

    ax.set_xlabel(r"True angle $\theta$ [rad]")
    ax.set_ylabel("Absolute error [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
