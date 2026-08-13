"""Linearization and discretization of the robot dynamics.

The robot is nonlinear. Iterative Learning Control needs a linear model along the path.
The functions here make that model:

  1. Write the dynamics in the state-space form with x = [th, th_d].
  2. Take the Jacobian against the state and against the torque. This gives A, B, C, D at
     one point of the path.
  3. Discretize with a zero-order hold, so that the model gives the state at the next
     time step.

Step 2 uses forward-mode automatic differentiation, so the code needs no hand-written
derivative of the equation of motion.
"""

from functools import partial
from typing import Callable, Dict, Tuple

from jax import Array, jacfwd, jit, vmap
from jax import numpy as jnp
from jax.scipy import linalg

from jax_double_pendulum.dynamics import (
    continuous_forward_dynamics,
    continuous_inverse_dynamics,
)
from robot_control.controllers import ctrl_fb_pd


@partial(jit, static_argnums=0, static_argnames=("continuous_forward_dynamics_fn",))
def continuous_state_space_dynamics(
    continuous_forward_dynamics_fn: Callable,
    x: Array,
    tau: Array,
    *args_dynamics,
) -> Tuple[Array, Array]:
    """Give dx/dt and the output y = th, for the state x = [th, th_d]."""
    th, th_d = x[:2], x[2:]
    th_dd = continuous_forward_dynamics_fn(th, th_d, tau, *args_dynamics)
    return jnp.concatenate([th_d, th_dd]), th


@partial(jit, static_argnums=0, static_argnames=("continuous_forward_dynamics_fn",))
def continuous_linear_state_space_representation_autograd(
    continuous_forward_dynamics_fn: Callable,
    th_eq: Array,
    th_d_eq: Array = jnp.zeros((2,)),
    tau_eq: Array = jnp.zeros((2,)),
    *args_dynamics,
) -> Tuple[Array, Array, Array, Array]:
    """Give A, B, C and D at an operating point, from automatic differentiation.

    Shapes: A (4, 4), B (4, 2), C (2, 4), D (2, 2).
    """
    x_eq = jnp.concatenate([th_eq, th_d_eq])

    fn = partial(continuous_state_space_dynamics, continuous_forward_dynamics_fn)
    (A, B), (C, D) = jacfwd(fn, argnums=(0, 1))(x_eq, tau_eq, *args_dynamics)

    return A, B, C, D


@jit
def cont2discrete_zoh(
    dt: Array, A: Array, B: Array, C: Array, D: Array
) -> Tuple[Array, Array, Array, Array]:
    """Discretize a state-space system with a zero-order hold.

    The method builds the block matrix [[A, B], [0, 0]] and takes its matrix exponential.
    The upper blocks of the result are Ad and Bd. This is exact for a torque that is
    constant over the time step.
    """
    n, m = A.shape[0], B.shape[1]

    em = jnp.concatenate(
        [jnp.concatenate([A, B], axis=1), jnp.zeros((m, n + m))], axis=0
    )
    ms = linalg.expm(em * dt)

    return ms[:n, :n], ms[:n, n:], C, D


@jit
def linearized_discrete_forward_dynamics(
    Ad: Array,
    Bd: Array,
    Cd: Array,
    Dd: Array,
    th_eq: Array,
    th_d_eq: Array,
    tau_eq: Array,
    dt: float,
    th: Array,
    th_d: Array,
    tau: Array,
) -> Tuple[Array, Array, Array]:
    """Make one step with the linear model.

    The model works on the difference against the operating point, so the code adds the
    operating point back to the result.
    """
    x_eq = jnp.concatenate([th_eq, th_d_eq])
    x = jnp.concatenate([th, th_d])

    delta_x_next = Ad @ (x - x_eq) + Bd @ (tau - tau_eq)
    x_next = x_eq + delta_x_next

    th_next, th_d_next = x_next[:2], x_next[2:]
    th_dd = (th_d_next - th_d) / dt

    return th_next, th_d_next, th_dd


def closed_loop_fb_continuous_forward_dynamics(
    rp: Dict,
    th: Array,
    th_d: Array,
    tau_ext: Array,
    th_des: Array,
    th_d_des: Array,
    kp_fb: Array = jnp.zeros((2,)),
    kd_fb: Array = jnp.zeros((2,)),
) -> Array:
    """The dynamics of the robot together with its PD feedback controller.

    ILC learns a correction for the closed loop, not for the robot alone. The controller
    must therefore be part of the model that is linearized.
    """
    tau_fb = ctrl_fb_pd(th, th_d, th_des, th_d_des, kp_fb, kd_fb)
    return continuous_forward_dynamics(rp, th, th_d, tau_ext + tau_fb)


def linearize_closed_loop_fb_system_about_trajectory(
    rp: Dict,
    traj_ts: Dict[str, Array],
    kp_fb: Array = jnp.zeros((2, 2)),
    kd_fb: Array = jnp.zeros((2, 2)),
) -> Tuple[Array, Array, Array, Array, Array]:
    """Linearize the closed loop at every time step of the path.

    The function is not jitted. `expm` runs under `vmap`, and that combination compiles
    slowly.

    Returns:
        tau_eq_ts (N, 2), Ad_ts (N, 4, 4), Bd_ts (N, 4, 2), Cd_ts (N, 2, 4),
        Dd_ts (N, 2, 2)
    """
    # The inverse dynamics give the torque that holds the path with a perfect model.
    tau_eq_ts = vmap(partial(continuous_inverse_dynamics, rp))(
        traj_ts["th_ts"], traj_ts["th_d_ts"], traj_ts["th_dd_ts"]
    )

    closed_loop_fn = partial(
        closed_loop_fb_continuous_forward_dynamics, rp, kp_fb=kp_fb, kd_fb=kd_fb
    )
    cl_lsp_autograd_fn = partial(
        continuous_linear_state_space_representation_autograd, closed_loop_fn
    )

    A_ts, B_ts, C_ts, D_ts = vmap(cl_lsp_autograd_fn)(
        traj_ts["th_ts"],
        traj_ts["th_d_ts"],
        tau_eq_ts,
        traj_ts["th_ts"],
        traj_ts["th_d_ts"],
    )

    dt = jnp.mean(traj_ts["t_ts"][1:] - traj_ts["t_ts"][:-1])
    Ad_ts, Bd_ts, Cd_ts, Dd_ts = vmap(partial(cont2discrete_zoh, dt))(
        A_ts, B_ts, C_ts, D_ts
    )

    return tau_eq_ts, Ad_ts, Bd_ts, Cd_ts, Dd_ts
