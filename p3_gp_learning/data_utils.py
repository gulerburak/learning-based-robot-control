"""Helpers that make a table of simulation data and turn it into training tensors."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from jax import numpy as jnp

DATA_DIR = Path("data") / "p3"
OUTPUT_DIR = Path("outputs") / "p3"

# The columns that hold no useful input or label for the learning tasks.
DROPPED_COLUMNS = [
    "tau_ff_ts_1",
    "tau_ff_ts_2",
    "x_d_ts_1",
    "x_d_ts_2",
    "x_dd_ts_1",
    "x_dd_ts_2",
    "x_eb_ts_1",
    "x_eb_ts_2",
    "x_ts_1",
    "x_ts_2",
]


def wrap_angle(angle):
    """Put an angle into the range [-pi, pi]."""
    return np.angle(np.exp(1j * angle))


def split_2d_columns(data: Dict) -> Dict:
    """Make one column for each component of a 2-column array.

    A DataFrame holds scalars, so `th_ts` of shape (N, 2) becomes `th_ts_1` and
    `th_ts_2`.
    """
    new_data = {}
    for key, value in data.items():
        if isinstance(value, jnp.ndarray) and value.ndim == 2:
            new_keys = [f"{key}_{i + 1}" for i in range(value.shape[1])]
            new_data.update(zip(new_keys, value.T))
        else:
            new_data[key] = value
    return new_data


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a simulation table and add the joint-space columns.

    The first row and the last row are removed, because their acceleration is not
    correct. The angles are wrapped, so that the periodic kernel sees a continuous input.
    """
    df = df.iloc[1:-1]
    df = df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns])

    df["th_ts_2_rel"] = wrap_angle(df["th_ts_2"] - df["th_ts_1"])
    df["th_d_ts_2_rel"] = df["th_d_ts_2"] - df["th_d_ts_1"]
    df["th_ts_2"] = wrap_angle(df["th_ts_2"])
    df["th_ts_1"] = wrap_angle(df["th_ts_1"])

    return df


def plot_data(
    df: pd.DataFrame,
    input_columns: List[str],
    output_columns: List[str],
    filepath: Optional[str] = None,
):
    """Plot every input and every label against the time."""
    columns = input_columns + output_columns
    rows = int(np.ceil(len(columns) / 2))

    fig, axes = plt.subplots(rows, 2, figsize=(10, 6), constrained_layout=True)
    fig.suptitle("Training inputs and labels")
    axes = np.atleast_2d(axes)

    for index, column in enumerate(columns):
        ax = axes[index // 2, index % 2]
        ax.set_title(column)
        ax.plot(df["t_ts"], df[column], color="lightblue")

    if filepath is not None:
        fig.savefig(filepath)
    plt.close(fig)


def generate_training_data(
    df: pd.DataFrame, input_cols: List[str], output_cols: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make the input tensor and the label tensor from a table."""
    train_x = torch.tensor(df[input_cols].to_numpy(), dtype=torch.float32)
    train_y = torch.tensor(df[output_cols].to_numpy(), dtype=torch.float32)

    if len(output_cols) == 1:
        train_y = train_y.unsqueeze(1)

    return train_x.contiguous(), train_y.contiguous()
