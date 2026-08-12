"""PD+ control that uses the learned M, C and G instead of the physical model.

This is the goal of the whole learning step. The controller keeps its structure, and only
the model in the feedforward term changes. The result is almost the same as the result
with the true model, so the network learned the physics and not only a fit of the data.

Run `python -m p2_control.tasks.train_lnn` first.

    python -m p2_control.tasks.control_with_lnn
"""

import argparse
from functools import partial

from jax import numpy as jnp

from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from p2_control.common import (
    initial_state,
    make_ellipse_trajectory,
    report_tracking_error,
    save_tracking_plots,
)
from p2_control.controllers import ctrl_fb_pd, ctrl_ff_pd_plus
from p2_control.lnn import dynamical_matrices as lnn_dynamical_matrices
from p2_control.tasks.rollout_lnn import load_nn_params

KP = 500.0
KD = 50.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kp", type=float, default=KP)
    parser.add_argument("--kd", type=float, default=KD)
    args = parser.parse_args()

    nn_params = load_nn_params()
    kp, kd = args.kp * jnp.eye(2), args.kd * jnp.eye(2)

    t_ts, traj_ts = make_ellipse_trajectory()
    th_0, th_d_0 = initial_state(traj_ts)

    lnn_dynamical_matrices_fn = partial(
        lnn_dynamical_matrices,
        nn_params["MassMatrixNN"],
        nn_params["PotentialEnergyNN"],
    )

    sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=th_0,
        th_d_0=th_d_0,
        th_des_ts=traj_ts["th_ts"],
        th_d_des_ts=traj_ts["th_d_ts"],
        th_dd_des_ts=traj_ts["th_dd_ts"],
        ctrl_fb=partial(ctrl_fb_pd, kp=kp, kd=kd),
        ctrl_ff=partial(ctrl_ff_pd_plus, lnn_dynamical_matrices_fn),
    )

    report_tracking_error(traj_ts, sim_ts, "PD+ with the learned dynamics")
    save_tracking_plots(traj_ts, sim_ts, "2c-5_pd_plus_learned")


if __name__ == "__main__":
    main()
