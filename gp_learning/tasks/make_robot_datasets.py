"""Make the three robot datasets that the learning tasks of project 3 need.

Run this script first.

The three datasets cover different parts of the state space. That difference is the point
of the comparison in the later tasks:

  1. `small_oscillation` — the robot hangs down and swings a little. It covers a small
     area around -pi/2 only.
  2. `big_oscillation`   — the robot starts near the upper position and swings over the
     full circle. It covers the whole state space.
  3. `pd_tracking`       — the robot follows an ellipse with a PD controller and gravity
     compensation. The feedback torques of this run are the labels for behavioural
     cloning.

    python -m gp_learning.tasks.make_robot_datasets
"""

import argparse
from functools import partial

import numpy as np
import pandas as pd
from jax import numpy as jnp

from jax_double_pendulum.dynamics import dynamical_matrices
from jax_double_pendulum.motion_planning import (
    ELLIPSE_PARAMS,
    generate_ellipse_trajectory,
)
from jax_double_pendulum.robot_parameters import ROBOT_PARAMS
from jax_double_pendulum.robot_simulation import simulate_robot
from robot_control.controllers import ctrl_fb_pd_rel, ctrl_ff_gravity_compensation
from gp_learning.data_utils import DATA_DIR, process_data, split_2d_columns

SIM_DT = 1e-2

KP_TRACKING = 2000.0
KD_TRACKING = 100.0


def to_dataframe(sim_ts) -> pd.DataFrame:
    return pd.DataFrame(split_2d_columns(dict(sim_ts)))


def make_free_motion_dataset(name: str, th_0, duration: float) -> pd.DataFrame:
    """Simulate the robot without a controller."""
    t_ts = SIM_DT * jnp.arange(int(duration / SIM_DT))
    sim_ts = simulate_robot(
        rp=ROBOT_PARAMS, t_ts=t_ts, th_0=th_0, th_d_0=jnp.array([0.0, 0.0])
    )
    print(f"{name}: {len(t_ts)} steps over {duration} s.")
    return process_data(to_dataframe(sim_ts))


def make_tracking_dataset(name: str, duration: float) -> pd.DataFrame:
    """Simulate the robot with a PD controller and gravity compensation."""
    t_ts = SIM_DT * jnp.arange(int(duration / SIM_DT))
    traj_ts = generate_ellipse_trajectory(rp=ROBOT_PARAMS, t_ts=t_ts, **ELLIPSE_PARAMS)

    kp = KP_TRACKING * jnp.eye(2)
    kd = KD_TRACKING * jnp.eye(2)

    sim_ts = simulate_robot(
        rp=ROBOT_PARAMS,
        t_ts=t_ts,
        th_0=traj_ts["th_ts"][0],
        th_d_0=traj_ts["th_d_ts"][0],
        th_des_ts=traj_ts["th_ts"],
        th_d_des_ts=traj_ts["th_d_ts"],
        th_dd_des_ts=traj_ts["th_dd_ts"],
        ctrl_ff=partial(
            ctrl_ff_gravity_compensation, partial(dynamical_matrices, ROBOT_PARAMS)
        ),
        ctrl_fb=partial(ctrl_fb_pd_rel, kp=kp, kd=kd),
    )

    df = to_dataframe(sim_ts)
    reference = np.array(traj_ts["th_ts"])
    reference_speed = np.array(traj_ts["th_d_ts"])
    df["ref_th_1"], df["ref_th_2"] = reference[:, 0], reference[:, 1]
    df["ref_th_d_1"], df["ref_th_d_2"] = reference_speed[:, 0], reference_speed[:, 1]

    print(f"{name}: {len(t_ts)} steps over {duration} s.")
    return process_data(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DATA_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "small_oscillation": make_free_motion_dataset(
            "small_oscillation", jnp.array([-jnp.pi / 2 + 0.1, -jnp.pi / 2 + 0.1]), 15.0
        ),
        "big_oscillation": make_free_motion_dataset(
            "big_oscillation", jnp.array([0.5 * jnp.pi + 0.1, 0.5 * jnp.pi - 0.1]), 50.0
        ),
        "pd_tracking": make_tracking_dataset("pd_tracking", 15.0),
    }

    for name, df in datasets.items():
        filepath = output_dir / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"Wrote {filepath} with {len(df)} rows.")


if __name__ == "__main__":
    main()
