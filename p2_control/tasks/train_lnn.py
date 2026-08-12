"""Train the Lagrangian Neural Network on the collected dataset.

Run `python -m p2_control.tasks.collect_dataset` first.

    python -m p2_control.tasks.train_lnn
    python -m p2_control.tasks.train_lnn --epochs 5     # a quick check

The script saves the parameters of the epoch with the smallest validation loss.
"""

import argparse

import dill
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random

from p2_control.common import CHECKPOINT_DIR, DATASET_DIR, OUTPUT_DIR
from p2_control.lnn_training import load_datasets, run_lnn_training
from p2_control.tasks.collect_dataset import DATASET_NAME

NUM_EPOCHS = 250
BASE_LR = 7e-4
WARMUP_EPOCHS = 10
BATCH_SIZE = 250
WEIGHT_DECAY = 0.0

PARAMS_NAME = "lagrangian_nn_params.pkl"


def plot_convergence(val_loss_history, train_metrics_history, filepath):
    train_loss = [m["loss"] for m in train_metrics_history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(train_loss, label="training loss")
    ax.semilogy(val_loss_history, label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"MSE of $\dot{\theta}$ at the next step")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--base-lr", type=float, default=BASE_LR)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.PRNGKey(seed=args.seed)
    rng, dataset_split_rng = random.split(rng)

    train_ds, val_ds = load_datasets(DATASET_DIR / DATASET_NAME, dataset_split_rng)
    print(
        f"Training samples: {len(train_ds['th_curr_ss'])}, "
        f"validation samples: {len(val_ds['th_curr_ss'])}."
    )

    val_loss_history, train_metrics_history, val_metrics_history, states_history = (
        run_lnn_training(
            rng,
            train_ds,
            val_ds,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            warmup_epochs=min(args.warmup_epochs, args.epochs),
            weight_decay=args.weight_decay,
        )
    )

    best_epoch = int(jnp.argmin(val_loss_history))
    best_metrics = val_metrics_history[best_epoch]
    print(
        f"Best epoch: {best_epoch + 1}. Validation loss: {val_loss_history[best_epoch]:.4e}. "
        f"RMSE of the next speed: {best_metrics['rmse_th_d_next']:.4e} rad/s."
    )

    states = states_history[best_epoch]
    nn_params = {
        "MassMatrixNN": states["MassMatrixNN"].params,
        "PotentialEnergyNN": states["PotentialEnergyNN"].params,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_DIR / PARAMS_NAME, "wb") as f:
        dill.dump(nn_params, f)
    print(f"Wrote {CHECKPOINT_DIR / PARAMS_NAME}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_convergence(
        val_loss_history, train_metrics_history, OUTPUT_DIR / "2c-3_lnn_convergence.pdf"
    )


if __name__ == "__main__":
    main()
