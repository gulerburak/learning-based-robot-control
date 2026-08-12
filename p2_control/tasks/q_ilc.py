"""Q-ILC against a wrong robot model.

The learning gains come from an LQR problem, so only the two weights Q and S must be
chosen. Q-ILC converges faster than PD-ILC and reaches a smaller error.

The lifted matrix P has a size of 1998 x 1998 and needs some minutes. The script writes
it into a cache file and reads it again in the next run.

    python -m p2_control.tasks.q_ilc
    python -m p2_control.tasks.q_ilc --iterations 20 --duration 2.0    # a quick check
"""

import argparse

from jax import numpy as jnp

from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from p2_control.common import (
    CACHE_DIR,
    OUTPUT_DIR,
    SIM_DURATION,
    initial_state,
    make_ellipse_trajectory,
    perturb_robot_params,
    report_tracking_error,
    save_tracking_plots,
)
from p2_control.controllers import ctrl_fb_pd
from p2_control.ilc_analysis import plot_configuration_space_ilc_convergence
from p2_control.ilc_q import compute_lifted_system_input_to_output_mapping, run_q_ilc
from p2_control.linearization import linearize_closed_loop_fb_system_about_trajectory

NUM_ITERATIONS = 1000
Q_WEIGHT = 1e0
S_WEIGHT = 5e-4

KP_FB = 500.0
KD_FB = 50.0
PERTURBATION_FACTOR = 1.8


def load_or_compute_lifted_system(traj_ts, rp_perturbed, kp_fb, kd_fb, use_cache=True):
    """Give the equilibrium torque and the lifted matrix P, from the cache if possible."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"q_ilc_N{traj_ts['t_ts'].shape[0]}.npz"

    if use_cache and cache_file.is_file():
        print(f"Read the cache {cache_file}.")
        cached = jnp.load(cache_file)
        return cached["tau_eq_ts"], cached["P"]

    tau_eq_ts, Ad_ts, Bd_ts, Cd_ts, Dd_ts = (
        linearize_closed_loop_fb_system_about_trajectory(
            rp_perturbed, traj_ts, kp_fb=kp_fb, kd_fb=kd_fb
        )
    )
    P = compute_lifted_system_input_to_output_mapping(Ad_ts, Bd_ts, Cd_ts, Dd_ts)

    jnp.savez(file=str(cache_file), tau_eq_ts=tau_eq_ts, P=P)
    print(f"Wrote the cache {cache_file}.")
    return tau_eq_ts, P


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--duration", type=float, default=SIM_DURATION)
    parser.add_argument("--q-weight", type=float, default=Q_WEIGHT)
    parser.add_argument("--s-weight", type=float, default=S_WEIGHT)
    parser.add_argument("--perturbation", type=float, default=PERTURBATION_FACTOR)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    kp_fb, kd_fb = KP_FB * jnp.eye(2), KD_FB * jnp.eye(2)
    t_ts, traj_ts = make_ellipse_trajectory(duration=args.duration)
    th_0, th_d_0 = initial_state(traj_ts)

    rp_perturbed = perturb_robot_params(args.perturbation)
    print(
        f"The controller model is wrong. The masses and the inertias are "
        f"{args.perturbation} times too large."
    )

    tau_eq_ts, P = load_or_compute_lifted_system(
        traj_ts, rp_perturbed, kp_fb, kd_fb, use_cache=not args.no_cache
    )

    num_lifted = P.shape[0]
    Q_lq = args.q_weight * jnp.eye(num_lifted)
    S_lq = args.s_weight * jnp.eye(P.shape[1])
    print(f"The lifted system has a size of {P.shape[0]} x {P.shape[1]}.")

    ilc_its = run_q_ilc(
        rp=ROBOT_PARAMS,
        traj_ts=traj_ts,
        th_0=th_0,
        th_d_0=th_d_0,
        num_iterations=args.iterations,
        tau_eq_ts=tau_eq_ts,
        P=P,
        Q_lq=Q_lq,
        S_lq=S_lq,
        kp_fb=kp_fb,
        kd_fb=kd_fb,
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

    report_tracking_error(traj_ts, sim_ts, f"Q-ILC after {args.iterations} iterations")
    save_tracking_plots(traj_ts, sim_ts, "2d-3_q_ilc")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_configuration_space_ilc_convergence(
        traj_ts, ilc_its, filepath=str(OUTPUT_DIR / "2d-3_q_ilc_convergence.pdf")
    )


if __name__ == "__main__":
    main()
