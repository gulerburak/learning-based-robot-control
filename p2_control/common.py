"""Settings and helpers that all control tasks use."""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from jax import Array
from jax import numpy as jnp

from jax_double_pendulum.analysis import (
    compute_configuration_space_rmse,
    compute_operational_space_rmse,
    plot_actuation,
    plot_configuration_space_trajectory_following,
    plot_operational_space_trajectory_following,
)
from jax_double_pendulum.motion_planning import (
    ELLIPSE_PARAMS,
    generate_ellipse_trajectory,
)
from jax_double_pendulum.robot_parameters import ROBOT_PARAMS

SIM_DT = 1e-2
SIM_DURATION = 10.0

# The robot starts away from the path, so that the feedback term must work.
INITIAL_OFFSET = jnp.array([0.1, 0.2])

OUTPUT_DIR = Path("outputs") / "p2"
DATASET_DIR = Path("data") / "p2"
CHECKPOINT_DIR = Path("outputs") / "p2" / "checkpoints"
CACHE_DIR = Path("outputs") / "p2" / "cache"


def make_time_steps(duration: float = SIM_DURATION, dt: float = SIM_DT) -> Array:
    return dt * jnp.arange(int(duration / dt))


def make_ellipse_trajectory(
    duration: float = SIM_DURATION, dt: float = SIM_DT
) -> Tuple[Array, Dict[str, Array]]:
    """Give the time steps and the reference path in the operational space."""
    t_ts = make_time_steps(duration, dt)
    traj_ts = generate_ellipse_trajectory(rp=ROBOT_PARAMS, t_ts=t_ts, **ELLIPSE_PARAMS)
    return t_ts, traj_ts


def initial_state(traj_ts: Dict[str, Array]) -> Tuple[Array, Array]:
    """Give the start angles and the start speeds, with the offset."""
    return traj_ts["th_ts"][0] - INITIAL_OFFSET, traj_ts["th_d_ts"][0]


def perturb_robot_params(factor: float, rp: Dict = ROBOT_PARAMS) -> Dict:
    """Make a wrong robot model. The masses and the inertias become larger."""
    rp_perturbed = dict(rp)
    for key in ("m1", "j1", "m2", "j2"):
        rp_perturbed[key] = factor * rp[key]
    return rp_perturbed


def report_tracking_error(
    traj_ts: Dict[str, Array], sim_ts: Dict[str, Array], name: str = ""
) -> float:
    """Print the tracking errors and give the norm of the operational-space RMSE."""
    rmse_th, rmse_th_d, _ = compute_configuration_space_rmse(traj_ts, sim_ts)
    rmse_x, rmse_x_d, _ = compute_operational_space_rmse(traj_ts, sim_ts)
    norm_rmse_x = float(jnp.linalg.norm(rmse_x))

    if name:
        print(f"--- {name} ---")
    print(f"RMSE of the link angles:  {rmse_th} rad")
    print(f"RMSE of the link speeds:  {rmse_th_d} rad/s")
    print(f"RMSE of the end position: {rmse_x} m")
    print(f"Norm of the position RMSE: {norm_rmse_x:.4f} m")
    return norm_rmse_x


def save_tracking_plots(
    traj_ts: Dict[str, Array],
    sim_ts: Dict[str, Array],
    prefix: str,
    output_dir: Path = OUTPUT_DIR,
):
    """Write the three standard figures of a tracking run.

    The plot helpers of the course make figures with a fixed name. A second call fails
    if the first figure is still open, so each figure is closed after it is written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_configuration_space_trajectory_following(
        traj_ts, sim_ts, filepath=str(output_dir / f"{prefix}_configuration_space.pdf")
    )
    plt.close("all")
    plot_operational_space_trajectory_following(
        traj_ts, sim_ts, filepath=str(output_dir / f"{prefix}_operational_space.pdf")
    )
    plt.close("all")
    plot_actuation(sim_ts, filepath=str(output_dir / f"{prefix}_actuation.pdf"))
    plt.close("all")
    print(f"Figures are in {output_dir}.")
