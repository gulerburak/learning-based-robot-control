"""Compare a rollout of the learned dynamics against the true dynamics.

The robot starts at rest and receives no torque. Both models then simulate the free
motion for 10 s. A learned model that keeps the energy follows the true motion for a long
time. A model that does not keep the energy diverges quickly.

Run `python -m robot_control.tasks.train_lnn` first.

    python -m robot_control.tasks.rollout_lnn
"""

import argparse
from functools import partial

import dill
from jax import numpy as jnp

from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from robot_control.common import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    make_time_steps,
    report_tracking_error,
)
from robot_control.lnn import discrete_forward_dynamics
from robot_control.tasks.train_lnn import PARAMS_NAME


def load_nn_params(filepath=None):
    filepath = filepath or CHECKPOINT_DIR / PARAMS_NAME
    with open(filepath, "rb") as f:
        return dill.load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()

    nn_params = load_nn_params()
    t_ts = make_time_steps(args.duration)
    th_0, th_d_0 = jnp.zeros((2,)), jnp.zeros((2,))

    nominal_sim_ts = simulate_robot(
        rp=ROBOT_PARAMS, t_ts=t_ts, th_0=th_0, th_d_0=th_d_0
    )

    learned_fn = partial(
        discrete_forward_dynamics,
        nn_params["MassMatrixNN"],
        nn_params["PotentialEnergyNN"],
    )
    learned_sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=th_0,
        th_d_0=th_d_0,
        discrete_forward_dynamics_fn=learned_fn,
    )

    report_tracking_error(nominal_sim_ts, learned_sim_ts, "Learned dynamics rollout")

    if args.animate:
        from jax_double_pendulum.visualization import animate_robot

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        animate_robot(
            ROBOT_PARAMS,
            sim_ts=nominal_sim_ts,
            sim_hat_ts=learned_sim_ts,
            step_skip=5,
            show=False,
            filepath=str(OUTPUT_DIR / "rollout_learned_dynamics.mp4"),
        )


if __name__ == "__main__":
    main()
