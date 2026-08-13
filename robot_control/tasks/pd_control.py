"""PD control of the robot, in the link space and in the joint space.

The two controllers use the same gains. The joint-space controller is better, because the
torque acts on the joints. The link-space controller lets link 2 oscillate, because the
dynamics of link 2 are faster than the dynamics of link 1.

    python -m robot_control.tasks.pd_control
"""

import argparse
from functools import partial

from jax import numpy as jnp

from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from robot_control.common import (
    initial_state,
    make_ellipse_trajectory,
    report_tracking_error,
    save_tracking_plots,
)
from robot_control.controllers import ctrl_fb_pd, ctrl_fb_pd_rel

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

    for name, ctrl_fb in (("link angles", ctrl_fb_pd), ("joint angles", ctrl_fb_pd_rel)):
        sim_ts = simulate_robot(
            rp=ROBOT_PARAMS,
            t_ts=t_ts,
            th_0=th_0,
            th_d_0=th_d_0,
            th_des_ts=traj_ts["th_ts"],
            th_d_des_ts=traj_ts["th_d_ts"],
            th_dd_des_ts=traj_ts["th_dd_ts"],
            ctrl_fb=partial(ctrl_fb, kp=kp, kd=kd),
        )
        report_tracking_error(traj_ts, sim_ts, f"PD on the {name}")
        save_tracking_plots(traj_ts, sim_ts, f"pd_{name.split()[0]}")


if __name__ == "__main__":
    main()
