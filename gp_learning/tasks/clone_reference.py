"""Clone a reference path with a Gaussian process, and control the robot with it.

Here the controller does not see the torque of a teacher. It sees the reference path
only. The GP learns the map

    (angles now) -> (change of the angles over one time step)

The change points along the path. A proportional gain makes a torque from it.

The script runs two closed loops:

  1. The pure cloned policy. The robot leaves the path. There is no speed term and no
     gravity compensation.
  2. The same policy plus gravity compensation, a damping term, and a term that moves the
     robot away from uncertainty. The error becomes smaller, but this controller stays
     much weaker than the controller of `clone_torques`. The change of the angle over one
     time step is a small signal against the gravity torque of the arm.

    python -m gp_learning.tasks.clone_reference
"""

import argparse
from functools import partial

import numpy as np
import torch
from jax import numpy as jnp

from jax_double_pendulum.dynamics import dynamical_matrices
from jax_double_pendulum.motion_planning import (
    ELLIPSE_PARAMS,
    generate_ellipse_trajectory,
)
from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from robot_control.controllers import ctrl_ff_gravity_compensation
from gp_learning.cloning import make_reference_cloning_controller
from gp_learning.data_utils import OUTPUT_DIR, wrap_angle
from gp_learning.gp_models import periodic_kernel
from gp_learning.tasks.clone_torques import plot_path, report_path_error
from gp_learning.wrappers import MultitaskGPRegressor

DATASET_DT = 1e-2
DATASET_DURATION = 15.0
STEP_AHEAD = 1

NUM_EPOCHS = 400
SEED = 42

KP = 500.0
KD = 2.0
K_VAR = 1.0

SIM_DT = 0.01
SIM_DURATION = 12.0


def make_delta_dataset():
    """Make the map from the angles now to the change of the angles.

    The second angle becomes a joint angle, and both angles are wrapped. The label is the
    difference to the next step of the path.
    """
    t_ts = DATASET_DT * jnp.arange(int(DATASET_DURATION / DATASET_DT))
    traj_ts = generate_ellipse_trajectory(rp=ROBOT_PARAMS, t_ts=t_ts, **ELLIPSE_PARAMS)

    angles = np.array(traj_ts["th_ts"])
    angles[:, 1] = angles[:, 1] - angles[:, 0]

    delta = angles[STEP_AHEAD:] - angles[:-STEP_AHEAD]
    angles = wrap_angle(angles[:-STEP_AHEAD])
    delta = wrap_angle(delta)

    X = torch.tensor(angles).float()
    Y = torch.tensor(delta).float()
    print(f"{len(X)} samples of the reference path.")
    return X, Y


def run_closed_loop(model, name: str, kp: float, kd: float, k_var: float, with_ff: bool):
    t_ts = SIM_DT * jnp.arange(int(SIM_DURATION / SIM_DT))
    traj_ts = generate_ellipse_trajectory(rp=ROBOT_PARAMS, t_ts=t_ts, **ELLIPSE_PARAMS)

    ctrl_fb = make_reference_cloning_controller(model, kp=kp, kd=kd, k_var=k_var)
    kwargs = {}
    if with_ff:
        kwargs["ctrl_ff"] = partial(
            ctrl_ff_gravity_compensation, partial(dynamical_matrices, ROBOT_PARAMS)
        )

    sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=traj_ts["th_ts"][0],
        th_d_0=traj_ts["th_d_ts"][0],
        ctrl_fb=ctrl_fb,
        jit_compile=False,
        **kwargs,
    )

    report_path_error(traj_ts, sim_ts, name)
    plot_path(traj_ts, sim_ts, name, OUTPUT_DIR / f"clone_reference_path_{name}.pdf")
    return sim_ts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--kp", type=float, default=KP)
    parser.add_argument("--kd", type=float, default=KD)
    parser.add_argument("--k-var", type=float, default=K_VAR)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, Y = make_delta_dataset()
    torch.manual_seed(SEED)
    model = MultitaskGPRegressor(X, Y, kernel_fn=periodic_kernel)
    model.train(num_epochs=args.epochs)
    model.plot_convergence(OUTPUT_DIR / "clone_reference_convergence.pdf")

    period = model.gp.covar_module.base_kernel.period_length
    print(f"Learned period of the kernel: {period.item():.4f} rad (2*pi = 6.2832)")

    run_closed_loop(
        model, "cloned_policy", kp=args.kp, kd=0.0, k_var=0.0, with_ff=False
    )
    run_closed_loop(
        model,
        "with_damping_and_variance",
        kp=args.kp,
        kd=args.kd,
        k_var=args.k_var,
        with_ff=True,
    )

    print(f"\nFigures are in {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
