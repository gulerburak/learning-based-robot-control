"""PD-type Iterative Learning Control.

ILC works when the robot repeats the same path many times. After each run the algorithm
changes the torque of the next run, so that the error of the last run becomes smaller:

    u_{j+1} = u_j + L * e_j + D * (e_j at k+1 minus e_j at k)

`L` and `D` come from the linear model of the closed loop. The learning is off-line: it
uses the full time series of one run, so it can also correct an error that occurs before
the time step that it changes.

The controller model is wrong on purpose. ILC must correct the model error from the data.
"""

from functools import partial
from typing import Dict, Tuple

from jax import Array, jit, lax, vmap
from jax import numpy as jnp

from robot_control.ilc import apply_ilc_control_action_to_system, init_ilc_its
from robot_control.linearization import linearize_closed_loop_fb_system_about_trajectory


@jit
def blk_diag(a: Array) -> Array:
    """Put the matrices of `a` on the diagonal of one large matrix.

    Args:
        a: shape (n, r, c)
    Returns:
        b: shape (n * r, n * c)
    """

    def assign_block_diagonal(i, _b):
        return lax.dynamic_update_slice(_b, a[i], (i * a.shape[1], i * a.shape[2]))

    b = jnp.zeros((a.shape[0] * a.shape[1], a.shape[0] * a.shape[2]))
    return lax.fori_loop(0, a.shape[0], assign_block_diagonal, b)


def compute_pd_ilc_gains(
    Bd_ts: Array, Cd_ts: Array, kp_ilc: float = 0.0, kd_ilc: float = 0.0
) -> Tuple[Array, Array]:
    """Give the learning matrices L and D.

    The inverse of (C_{k+1} @ B_k) maps an output error back to a torque. The gains are
    therefore correct for the units of the system, and only two scalars must be tuned.
    """
    inv_CB = vmap(lambda Cd_kp1, Bd_k: jnp.linalg.inv(Cd_kp1 @ Bd_k))(
        Cd_ts[1:], Bd_ts[:-1]
    )
    block = blk_diag(inv_CB)
    return kp_ilc * block, kd_ilc * block


@jit
def learning_rule_pd_ilc(
    u_ts: Array, y_ts: Array, y_des_ts: Array, L: Array, D: Array
) -> Array:
    """Give the torque correction of the next iteration.

    The time series become one long vector, which is the "lifted" form of the system.
    """
    N = u_ts.shape[0]

    Y = y_ts[1:].reshape(-1)
    Y_des = y_des_ts[1:].reshape(-1)
    E = Y_des - Y

    e_ts = y_des_ts - y_ts
    delta_E = (e_ts[1:] - e_ts[:-1]).reshape(-1)

    U = u_ts[:-1].reshape(-1)
    U_next = U + L @ E + D @ delta_E

    return jnp.zeros_like(u_ts).at[:-1].set(U_next.reshape((N - 1, 2)))


@jit
def pd_ilc_iteration(
    rp: Dict,
    traj_ts: Dict[str, Array],
    th_0: Array,
    th_d_0: Array,
    tau_eq_ts: Array,
    L: Array,
    D: Array,
    kp_fb: Array,
    kd_fb: Array,
    it: int,
    ilc_its: Dict[str, Array],
) -> Dict[str, Array]:
    """Run one iteration: apply the learnt torque, then improve it."""
    u_ts = ilc_its["u_nextit_ts"]
    ilc_its["u_its"] = ilc_its["u_its"].at[it].set(u_ts)

    tau_ilc_ts = tau_eq_ts + u_ts
    sim_ts, ilc_its = apply_ilc_control_action_to_system(
        rp=rp,
        traj_ts=traj_ts,
        th_0=th_0,
        th_d_0=th_d_0,
        it=it,
        ilc_its=ilc_its,
        tau_ilc_ts=tau_ilc_ts,
        kp_fb=kp_fb,
        kd_fb=kd_fb,
    )

    ilc_its["u_nextit_ts"] = learning_rule_pd_ilc(
        u_ts, sim_ts["th_ts"], traj_ts["th_ts"], L, D
    )
    return ilc_its


def run_pd_ilc(
    rp: Dict,
    traj_ts: Dict[str, Array],
    th_0: Array,
    th_d_0: Array,
    num_iterations: int,
    kp_ilc: float = 0.0,
    kd_ilc: float = 0.0,
    kp_fb: Array = jnp.zeros((2, 2)),
    kd_fb: Array = jnp.zeros((2, 2)),
    rp_perturbed: Dict = None,
) -> Dict[str, Array]:
    """Run the PD-ILC algorithm.

    Args:
        rp: the true robot parameters, used by the simulation
        rp_perturbed: the wrong parameters, used by the controller model
    """
    if rp_perturbed is None:
        rp_perturbed = rp.copy()

    tau_eq_ts, _, Bd_ts, Cd_ts, _ = linearize_closed_loop_fb_system_about_trajectory(
        rp_perturbed, traj_ts, kp_fb=kp_fb, kd_fb=kd_fb
    )

    ilc_its = init_ilc_its(num_iterations, traj_ts)
    ilc_its["tau_eq_ts"] = tau_eq_ts
    ilc_its["u_nextit_ts"] = jnp.zeros_like(traj_ts["th_ts"])

    L, D = compute_pd_ilc_gains(Bd_ts, Cd_ts, kp_ilc, kd_ilc)

    ilc_its = lax.fori_loop(
        0,
        num_iterations,
        partial(
            pd_ilc_iteration, rp, traj_ts, th_0, th_d_0, tau_eq_ts, L, D, kp_fb, kd_fb
        ),
        ilc_its,
    )
    ilc_its.pop("u_nextit_ts")
    return ilc_its
