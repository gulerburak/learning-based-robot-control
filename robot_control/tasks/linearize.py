"""Linearize the robot at the lower equilibrium and compare the two rollouts.

The linear model is correct near the operating point. The robot starts at the
equilibrium, so both models agree at the start. The nonlinear terms then grow, and the
two rollouts separate.

    python -m robot_control.tasks.linearize
"""

import argparse
from functools import partial

from jax import numpy as jnp

from jax_double_pendulum.dynamics import continuous_forward_dynamics, dynamical_matrices
from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from robot_control.common import SIM_DT, OUTPUT_DIR, make_time_steps
from robot_control.linearization import (
    cont2discrete_zoh,
    continuous_linear_state_space_representation_autograd,
    linearized_discrete_forward_dynamics,
)

DURATION = 4.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=DURATION)
    args = parser.parse_args()

    th_eq, th_d_eq = jnp.zeros((2,)), jnp.zeros((2,))

    # At the equilibrium the torque must hold the weight of the arm.
    _, _, G_eq = dynamical_matrices(ROBOT_PARAMS, th_eq, th_d_eq)
    tau_eq = G_eq

    forward_dynamics_fn = partial(continuous_forward_dynamics, ROBOT_PARAMS)
    A, B, C, D = continuous_linear_state_space_representation_autograd(
        forward_dynamics_fn, th_eq, th_d_eq, tau_eq
    )
    Ad, Bd, Cd, Dd = cont2discrete_zoh(SIM_DT, A, B, C, D)

    print("A =\n", A)
    print("B =\n", B)
    print("Ad =\n", Ad)
    print("Bd =\n", Bd)

    t_ts = make_time_steps(args.duration)
    nominal_sim_ts = simulate_robot(
        rp=ROBOT_PARAMS, t_ts=t_ts, th_0=th_eq, th_d_0=th_d_eq
    )
    linearized_sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=th_eq,
        th_d_0=th_d_eq,
        discrete_forward_dynamics_fn=partial(
            linearized_discrete_forward_dynamics, Ad, Bd, Cd, Dd, th_eq, th_d_eq, tau_eq
        ),
    )

    error = jnp.abs(linearized_sim_ts["th_ts"] - nominal_sim_ts["th_ts"]).max()
    print(f"Largest angle difference over {args.duration} s: {error:.3e} rad")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_comparison(nominal_sim_ts, linearized_sim_ts)


def _plot_comparison(nominal_sim_ts, linearized_sim_ts):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for index in range(2):
        ax.plot(
            nominal_sim_ts["t_ts"],
            nominal_sim_ts["th_ts"][:, index],
            label=rf"nonlinear $\theta_{index + 1}$",
        )
        ax.plot(
            linearized_sim_ts["t_ts"],
            linearized_sim_ts["th_ts"][:, index],
            "--",
            label=rf"linear $\theta_{index + 1}$",
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Link angle [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "linearization_rollout.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
