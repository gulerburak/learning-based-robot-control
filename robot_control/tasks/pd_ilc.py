"""PD-ILC against a wrong robot model.

The controller model has masses and inertias that are three times too large. A PD
controller with that model tracks the path badly. PD-ILC learns a torque correction over
many runs and removes most of the error, although the model stays wrong.

    python -m robot_control.tasks.pd_ilc
    python -m robot_control.tasks.pd_ilc --iterations 20    # a quick check
"""

import argparse

from jax import numpy as jnp

from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from robot_control.common import (
    OUTPUT_DIR,
    initial_state,
    make_ellipse_trajectory,
    perturb_robot_params,
    report_tracking_error,
    save_tracking_plots,
)
from robot_control.controllers import ctrl_fb_pd
from robot_control.ilc_analysis import plot_configuration_space_ilc_convergence
from robot_control.ilc_pd import run_pd_ilc

NUM_ITERATIONS = 500
KP_ILC = 2e-5
KD_ILC = 2e-3

KP_FB = 500.0
KD_FB = 50.0
PERTURBATION_FACTOR = 3.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--kp-ilc", type=float, default=KP_ILC)
    parser.add_argument("--kd-ilc", type=float, default=KD_ILC)
    parser.add_argument("--perturbation", type=float, default=PERTURBATION_FACTOR)
    args = parser.parse_args()

    kp_fb, kd_fb = KP_FB * jnp.eye(2), KD_FB * jnp.eye(2)
    t_ts, traj_ts = make_ellipse_trajectory()
    th_0, th_d_0 = initial_state(traj_ts)

    rp_perturbed = perturb_robot_params(args.perturbation)
    print(
        f"The controller model is wrong. The masses and the inertias are "
        f"{args.perturbation} times too large."
    )

    ilc_its = run_pd_ilc(
        rp=ROBOT_PARAMS,
        traj_ts=traj_ts,
        th_0=th_0,
        th_d_0=th_d_0,
        num_iterations=args.iterations,
        kp_ilc=args.kp_ilc,
        kd_ilc=args.kd_ilc,
        kp_fb=kp_fb,
        kd_fb=kd_fb,
        rp_perturbed=rp_perturbed,
    )

    sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=th_0,
        th_d_0=th_d_0,
        tau_ext_ts=ilc_its["tau_ilc_its"][-1],
        th_des_ts=traj_ts["th_ts"],
        th_d_des_ts=traj_ts["th_d_ts"],
        th_dd_des_ts=traj_ts["th_dd_ts"],
        ctrl_ff=lambda th, th_d, th_des, th_d_des, th_dd_des: jnp.zeros((2,)),
        ctrl_fb=lambda th, th_d, th_des, th_d_des: ctrl_fb_pd(
            th, th_d, th_des, th_d_des, kp_fb, kd_fb
        ),
    )

    report_tracking_error(traj_ts, sim_ts, f"PD-ILC after {args.iterations} iterations")
    save_tracking_plots(traj_ts, sim_ts, "pd_ilc")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_configuration_space_ilc_convergence(
        traj_ts, ilc_its, filepath=str(OUTPUT_DIR / "pd_ilc_convergence.pdf")
    )


if __name__ == "__main__":
    main()
