"""PD control with gravity compensation.

The feedforward term cancels the gravity torque at the measured state. The feedback term
therefore does not have to hold the weight of the arm, and the tracking error becomes
much smaller.

    python -m p2_control.tasks.pd_gravity_compensation
"""

import argparse
from functools import partial

from jax import numpy as jnp

from jax_double_pendulum.dynamics import dynamical_matrices
from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from p2_control.common import (
    initial_state,
    make_ellipse_trajectory,
    report_tracking_error,
    save_tracking_plots,
)
from p2_control.controllers import ctrl_fb_pd, ctrl_ff_gravity_compensation

KP = 5000.0
KD = 500.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kp", type=float, default=KP)
    parser.add_argument("--kd", type=float, default=KD)
    args = parser.parse_args()

    kp, kd = args.kp * jnp.eye(2), args.kd * jnp.eye(2)
    t_ts, traj_ts = make_ellipse_trajectory()
    th_0, th_d_0 = initial_state(traj_ts)

    ctrl_ff = partial(
        ctrl_ff_gravity_compensation, partial(dynamical_matrices, ROBOT_PARAMS)
    )

    sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=th_0,
        th_d_0=th_d_0,
        th_des_ts=traj_ts["th_ts"],
        th_d_des_ts=traj_ts["th_d_ts"],
        th_dd_des_ts=traj_ts["th_dd_ts"],
        ctrl_ff=ctrl_ff,
        ctrl_fb=partial(ctrl_fb_pd, kp=kp, kd=kd),
    )

    report_tracking_error(traj_ts, sim_ts, "PD with gravity compensation")
    save_tracking_plots(traj_ts, sim_ts, "2a-2_pd_gravity_compensation")


if __name__ == "__main__":
    main()
