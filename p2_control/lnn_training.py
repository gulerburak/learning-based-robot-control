"""The training pipeline of the Lagrangian Neural Network.

The loss compares the angular speed at the next time step. The prediction comes from one
RK4 step of the learned dynamics. The gradient therefore goes through the integrator and
through all derivatives of the Lagrangian.
"""

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import jax
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import Array, jit, random
from jax import numpy as jnp
from tqdm import tqdm

from jax_double_pendulum.utils import normalize_link_angles
from p2_control.lnn import MassMatrixNN, PotentialEnergyNN, discrete_forward_dynamics


def load_datasets(
    filepath: Path, rng: Array, val_ratio: float = 0.2
) -> Tuple[Dict[str, Array], Dict[str, Array]]:
    """Split the dataset into a training set and a validation set."""
    assert 0.0 <= val_ratio <= 1.0, "The validation ratio must be in [0, 1]."

    dataset = jnp.load(filepath)
    num_samples = dataset["th_curr_ss"].shape[0]

    perm = random.permutation(rng, num_samples)
    num_train = num_samples - int(val_ratio * num_samples)

    train_ds = {k: jnp.array(v)[perm[:num_train]] for k, v in dataset.items()}
    val_ds = {k: jnp.array(v)[perm[num_train:]] for k, v in dataset.items()}

    return train_ds, val_ds


def create_learning_rate_fn(
    num_epochs: int, steps_per_epoch: int, base_lr: float, warmup_epochs: int = 0
) -> Callable:
    """Make a schedule with a linear warmup and then a cosine decay."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_epochs * steps_per_epoch,
    )
    cosine_decay_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=(num_epochs - warmup_epochs) * steps_per_epoch,
    )
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_decay_fn],
        boundaries=[warmup_epochs * steps_per_epoch],
    )


def initialize_train_states(
    rng: Array, learning_rate_fn: Callable, weight_decay: float = 0.0
) -> Dict[str, TrainState]:
    """Make the parameters and the optimizers of the two networks."""
    mass_matrix_nn = MassMatrixNN()
    potential_energy_nn = PotentialEnergyNN()

    rng, rng_m, rng_u = random.split(rng, 3)
    mass_matrix_nn_params = mass_matrix_nn.init(rng_m, jnp.ones((2,)))["params"]
    potential_energy_nn_params = potential_energy_nn.init(rng_u, jnp.ones((2,)))[
        "params"
    ]

    tx_kwargs = dict(learning_rate=learning_rate_fn, weight_decay=weight_decay)

    return {
        "MassMatrixNN": TrainState.create(
            apply_fn=mass_matrix_nn.apply,
            params=mass_matrix_nn_params,
            tx=optax.adamw(**tx_kwargs),
        ),
        "PotentialEnergyNN": TrainState.create(
            apply_fn=potential_energy_nn.apply,
            params=potential_energy_nn_params,
            tx=optax.adamw(**tx_kwargs),
        ),
    }


@jit
def mse_loss_fn(pred: Array, target: Array) -> Array:
    return jnp.mean(jnp.square(pred - target))


@jit
def compute_metrics(batch: Dict[str, Array], preds: Dict[str, Array]) -> Dict[str, Array]:
    """Give the RMSE of the next angle and of the next angular speed.

    The angle error is wrapped to [-pi, pi], because an error of 2*pi is no error.
    """
    error_th = normalize_link_angles(preds["th_next_ss"] - batch["th_next_ss"])
    return {
        "rmse_th_next": jnp.sqrt(jnp.mean(jnp.square(error_th))),
        "rmse_th_d_next": jnp.sqrt(
            mse_loss_fn(preds["th_d_next_ss"], batch["th_d_next_ss"])
        ),
    }


def make_vectorized_discrete_forward_dynamics() -> Callable:
    """Map the one-sample dynamics over a batch. The two parameter sets are shared."""
    return jax.vmap(discrete_forward_dynamics, in_axes=(None, None, 0, 0, 0, 0))


@partial(jit, static_argnums=2, static_argnames="learning_rate_fn")
def train_step(
    states: Dict[str, TrainState], batch: Dict[str, Array], learning_rate_fn: Callable
) -> Tuple[Dict[str, TrainState], Dict[str, Array]]:
    """Make one gradient step for both networks."""

    def loss_fn(mass_matrix_nn_params: Dict, potential_energy_nn_params: Dict):
        vectorized_fn = make_vectorized_discrete_forward_dynamics()
        th_next_pred, th_d_next_pred, th_dd_pred = vectorized_fn(
            mass_matrix_nn_params,
            potential_energy_nn_params,
            batch["dt_ss"],
            batch["th_curr_ss"],
            batch["th_d_curr_ss"],
            batch["tau_ss"],
        )
        loss = mse_loss_fn(th_d_next_pred, batch["th_d_next_ss"])
        preds = {
            "th_next_ss": th_next_pred,
            "th_d_next_ss": th_d_next_pred,
            "th_dd_ss": th_dd_pred,
        }
        return loss, preds

    (loss, preds), (grad_mass_matrix_nn, grad_potential_energy_nn) = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(states["MassMatrixNN"].params, states["PotentialEnergyNN"].params)

    states["MassMatrixNN"] = states["MassMatrixNN"].apply_gradients(
        grads=grad_mass_matrix_nn
    )
    states["PotentialEnergyNN"] = states["PotentialEnergyNN"].apply_gradients(
        grads=grad_potential_energy_nn
    )

    metrics = compute_metrics(batch, preds)
    metrics["loss"] = loss
    metrics["lr_mass_matrix_nn"] = learning_rate_fn(states["MassMatrixNN"].step)
    metrics["lr_potential_energy_nn"] = learning_rate_fn(
        states["PotentialEnergyNN"].step
    )
    return states, metrics


@jit
def eval_step(states: Dict[str, TrainState], batch: Dict[str, Array]) -> Dict[str, Array]:
    vectorized_fn = make_vectorized_discrete_forward_dynamics()
    th_next_pred, th_d_next_pred, _ = vectorized_fn(
        states["MassMatrixNN"].params,
        states["PotentialEnergyNN"].params,
        batch["dt_ss"],
        batch["th_curr_ss"],
        batch["th_d_curr_ss"],
        batch["tau_ss"],
    )
    preds = {"th_next_ss": th_next_pred, "th_d_next_ss": th_d_next_pred}

    metrics = compute_metrics(batch, preds)
    metrics["loss"] = mse_loss_fn(th_d_next_pred, batch["th_d_next_ss"])
    return metrics


def train_epoch(
    states: Dict[str, TrainState],
    train_ds: Dict[str, Array],
    batch_size: int,
    learning_rate_fn: Callable,
    rng: Array,
) -> Tuple[Dict[str, TrainState], float, Dict[str, float]]:
    """Train over the full training set one time, in a random order."""
    train_ds_size = int(train_ds["th_curr_ss"].shape[0])
    steps_per_epoch = train_ds_size // batch_size

    perms = random.permutation(rng, train_ds_size)
    perms = perms[: steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))

    batch_metrics = []
    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        states, metrics = train_step(states, batch, learning_rate_fn)
        batch_metrics.append(metrics)

    batch_metrics = jax.device_get(batch_metrics)
    epoch_metrics = {
        k: np.mean(jnp.array([m[k] for m in batch_metrics])).item()
        for k in batch_metrics[0]
    }
    return states, epoch_metrics["loss"], epoch_metrics


def eval_model(
    states: Dict[str, TrainState], val_ds: Dict[str, Array]
) -> Tuple[float, Dict[str, float]]:
    val_metrics = jax.device_get(eval_step(states, val_ds))
    val_metrics = jax.tree_util.tree_map(lambda x: x.item(), val_metrics)
    return val_metrics["loss"], val_metrics


def run_lnn_training(
    rng: Array,
    train_ds: Dict[str, Array],
    val_ds: Dict[str, Array],
    num_epochs: int,
    batch_size: int,
    base_lr: float,
    warmup_epochs: int = 0,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> Tuple[Array, List[Dict], List[Dict], List[Dict]]:
    """Train the network and give the history of every epoch.

    The history holds the state of each epoch, so that the caller can select the epoch
    with the smallest validation loss.
    """
    num_train_samples = len(train_ds["th_curr_ss"])
    learning_rate_fn = create_learning_rate_fn(
        num_epochs, num_train_samples // batch_size, base_lr, warmup_epochs
    )

    rng, init_rng = random.split(rng, 2)
    states = initialize_train_states(init_rng, learning_rate_fn, weight_decay)

    val_loss_history, train_metrics_history = [], []
    val_metrics_history, states_history = [], []

    if verbose:
        print(f"Train the Lagrangian neural network for {num_epochs} epochs...")

    for epoch in (pbar := tqdm(range(1, num_epochs + 1))):
        rng, epoch_rng = random.split(rng)

        states, train_loss, train_metrics = train_epoch(
            states, train_ds, batch_size, learning_rate_fn, epoch_rng
        )
        val_loss, val_metrics = eval_model(states, val_ds)

        val_loss_history.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        states_history.append(states)

        if verbose:
            pbar.set_description(
                "Epoch: %d, lr: %.6f, train loss: %.7f, val loss: %.7f"
                % (epoch, train_metrics["lr_mass_matrix_nn"], train_loss, val_loss)
            )

    return jnp.array(val_loss_history), train_metrics_history, val_metrics_history, states_history
