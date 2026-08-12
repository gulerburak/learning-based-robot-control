"""Q-ILC: Iterative Learning Control with LQR-optimal learning gains.

PD-ILC needs two gains that a person must tune. Q-ILC computes the gain matrix from a
cost function instead:

    min over U of   E.T @ Q @ E + dU.T @ S @ dU

`Q` gives the weight of the tracking error, and `S` gives the weight of the change of the
torque. The solution is

    L_opt = (P.T @ Q @ P + S)^-1 @ P.T @ Q

`P` is the lifted-system matrix. It maps the full torque time series to the full output
time series, so `P[i, j]` shows the effect of the torque at step j on the output at step
i. `P` is block lower triangular, because a torque cannot change the past.

The matrices are large. With N = 1000 time steps and 2 inputs, `P` has a size of
1998 x 1998.
"""

from functools import partial
from typing import Dict

from jax import Array, jit, lax
from jax import numpy as jnp

from p2_control.ilc import apply_ilc_control_action_to_system, init_ilc_its


def compute_lifted_system_input_to_output_mapping(
    Ad_ts: Array, Bd_ts: Array, Cd_ts: Array, Dd_ts: Array
) -> Array:
    """Build the lifted matrix P of the linear time-varying system.

    The block at row i and column j is  C_{i+1} @ (A_i ... A_{j+1}) @ B_j.
    The loop keeps the product of the A matrices, so each block needs one extra
    multiplication only.
    """
    print("Compute the P matrix...")

    N = Ad_ts.shape[0]
    m = Ad_ts.shape[-1]
    n = Bd_ts.shape[-1]
    o = Cd_ts.shape[-2]

    P = jnp.zeros(((N - 1) * o, (N - 1) * n))

    def col_body(j, _P):
        _P = lax.dynamic_update_slice(_P, Cd_ts[j + 1] @ Bd_ts[j], (j * o, j * n))

        def inner_body(i, carry):
            A_prod, __P = carry
            A_prod = Ad_ts[i] @ A_prod
            block = Cd_ts[i + 1] @ A_prod @ Bd_ts[j]
            return A_prod, lax.dynamic_update_slice(__P, block, (i * o, j * n))

        _, _P = lax.fori_loop(j + 1, N - 1, inner_body, (jnp.eye(m), _P))
        return _P

    P = lax.fori_loop(0, N - 1, col_body, P)

    print("The P matrix is ready.")
    return P


def compute_lqr_optimal_gains(P: Array, Q_lq: Array = None, S_lq: Array = None) -> Array:
    """Give the optimal learning gain matrix L_opt."""
    if Q_lq is None:
        Q_lq = jnp.eye(P.shape[0])
    if S_lq is None:
        S_lq = jnp.eye(P.shape[1])

    assert Q_lq.shape[0] == P.shape[0], "Q_lq must match the number of outputs."
    assert S_lq.shape[0] == P.shape[1], "S_lq must match the number of inputs."

    return jnp.linalg.solve(P.T @ Q_lq @ P + S_lq, P.T @ Q_lq)


@jit
def learning_rule_q_ilc(
    u_ts: Array, y_ts: Array, y_des_ts: Array, L_opt: Array
) -> Array:
    """Give the torque correction of the next iteration."""
    N = u_ts.shape[0]

    E = y_des_ts[1:].reshape(-1) - y_ts[1:].reshape(-1)
    U = u_ts[:-1].reshape(-1)
    U_next = U + L_opt @ E

    return jnp.zeros_like(u_ts).at[:-1].set(U_next.reshape((N - 1, 2)))


@jit
def q_ilc_iteration(
    rp: Dict,
    traj_ts: Dict[str, Array],
    th_0: Array,
    th_d_0: Array,
    tau_eq_ts: Array,
    L_opt: Array,
    kp_fb: Array,
    kd_fb: Array,
    it: int,
    ilc_its: Dict[str, Array],
) -> Dict[str, Array]:
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

    ilc_its["u_nextit_ts"] = learning_rule_q_ilc(
        u_ts, sim_ts["th_ts"], traj_ts["th_ts"], L_opt
    )
    return ilc_its


def run_q_ilc(
    rp: Dict,
    traj_ts: Dict[str, Array],
    th_0: Array,
    th_d_0: Array,
    num_iterations: int,
    tau_eq_ts: Array,
    P: Array,
    Q_lq: Array = None,
    S_lq: Array = None,
    kp_fb: Array = jnp.zeros((2, 2)),
    kd_fb: Array = jnp.zeros((2, 2)),
) -> Dict[str, Array]:
    """Run the Q-ILC algorithm."""
    print("Compute the Q-ILC gains...")
    L_opt = compute_lqr_optimal_gains(P, Q_lq, S_lq)

    ilc_its = init_ilc_its(num_iterations, traj_ts)
    ilc_its["tau_eq_ts"] = tau_eq_ts
    ilc_its["u_nextit_ts"] = jnp.zeros_like(tau_eq_ts)
    ilc_its["L_opt"] = L_opt

    print("Run the Q-ILC algorithm...")
    ilc_its = lax.fori_loop(
        0,
        num_iterations,
        partial(
            q_ilc_iteration, rp, traj_ts, th_0, th_d_0, tau_eq_ts, L_opt, kp_fb, kd_fb
        ),
        ilc_its,
    )
    ilc_its.pop("u_nextit_ts")
    return ilc_its
