"""Collect a dataset of the robot dynamics for the Lagrangian Neural Network.

Each simulation starts at a random state and applies a random torque. The dataset holds
the transitions (th, th_d, tau) -> (th_next, th_d_next).

The two limits control the quality of the dataset:

  * The torque must be large. With a small torque the network cannot separate the
    potential energy from the inertia, because the two effects always occur together.
  * The torque must not be too large. If the torque is much larger than the gravity
    torque, the network learns the potential energy badly.

    python -m p2_control.tasks.collect_dataset
    python -m p2_control.tasks.collect_dataset --num-simulations 5    # a quick check
"""

import argparse
from typing import Dict, Tuple

from jax import Array, jit, lax, random
from jax import numpy as jnp
from tqdm import tqdm

from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from p2_control.common import DATASET_DIR, SIM_DT, make_time_steps

NUM_SIMULATIONS = 250
SIM_DURATION = 10.0
MAX_TH_D_0 = 2 * jnp.pi
MAX_TAU = 100.0

DATASET_NAME = "double_pendulum_dynamics.npz"


@jit
def sample_system_evolution(
    rng: Array, rp: Dict, t_ts: Array, max_th_d_0: Array, max_tau_ts: Array
) -> Tuple[Dict[str, Array], Array]:
    """Simulate the robot from a random state, with a random torque."""
    rng_updated, rng_th, rng_th_d, rng_tau = random.split(rng, 4)

    th_0 = random.uniform(rng_th, shape=(2,), minval=-jnp.pi, maxval=jnp.pi)
    th_d_0 = random.uniform(
        rng_th_d, shape=max_th_d_0.shape, minval=-max_th_d_0, maxval=max_th_d_0
    )
    tau_ts = random.uniform(
        rng_tau, shape=max_tau_ts.shape, minval=-max_tau_ts, maxval=max_tau_ts
    )

    sim_ts = simulate_robot(
        rp=rp, t_ts=t_ts, th_0=th_0, th_d_0=th_d_0, tau_ext_ts=tau_ts, jit_compile=True
    )
    return sim_ts, rng_updated


@jit
def save_sim_data_to_dataset(
    dataset: Dict[str, Array], sim_idx: int, sim_ts: Dict[str, Array]
) -> Dict[str, Array]:
    """Write the transitions of one simulation into the dataset arrays."""
    num_steps = sim_ts["t_ts"].shape[0]
    start_idx = sim_idx * (num_steps - 1)

    samples = {
        "tau_ss": sim_ts["tau_ts"][:-1],
        "th_curr_ss": sim_ts["th_ts"][:-1],
        "th_d_curr_ss": sim_ts["th_d_ts"][:-1],
        "th_next_ss": sim_ts["th_ts"][1:],
        "th_d_next_ss": sim_ts["th_d_ts"][1:],
    }
    for key, value in samples.items():
        dataset[key] = lax.dynamic_update_slice(dataset[key], value, (start_idx, 0))

    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-simulations", type=int, default=NUM_SIMULATIONS)
    parser.add_argument("--duration", type=float, default=SIM_DURATION)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t_ts = make_time_steps(args.duration, SIM_DT)
    max_th_d_0 = MAX_TH_D_0 * jnp.ones((2,))
    max_tau_ts = MAX_TAU * jnp.ones((t_ts.shape[0], 2))

    num_samples = args.num_simulations * (t_ts.shape[0] - 1)
    print(
        f"Run {args.num_simulations} simulations of {args.duration} s. "
        f"This gives {num_samples} samples."
    )

    dataset = {
        "dt_ss": SIM_DT * jnp.ones(num_samples),
        "tau_ss": jnp.zeros((num_samples, 2)),
        "th_curr_ss": jnp.zeros((num_samples, 2)),
        "th_d_curr_ss": jnp.zeros((num_samples, 2)),
        "th_next_ss": jnp.zeros((num_samples, 2)),
        "th_d_next_ss": jnp.zeros((num_samples, 2)),
    }

    rng = random.PRNGKey(seed=args.seed)
    for sim_idx in tqdm(range(args.num_simulations)):
        sim_ts, rng = sample_system_evolution(
            rng, ROBOT_PARAMS, t_ts, max_th_d_0, max_tau_ts
        )
        dataset = save_sim_data_to_dataset(dataset, sim_idx, sim_ts)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATASET_DIR / DATASET_NAME
    jnp.savez(file=str(filepath), **dataset)
    print(f"Wrote {filepath}.")


if __name__ == "__main__":
    main()
