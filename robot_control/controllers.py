"""Feedback and feedforward controllers for the 2-link robot.

The robot has two links. `th` holds the two absolute link angles, measured against the
vertical. `th_rel` holds the two joint angles: the first is the same as the first link
angle, and the second is the angle of link 2 against link 1.

A feedback controller has the signature `(th, th_d, th_des, th_d_des) -> tau`.
A feedforward controller has the signature
`(th, th_d, th_des, th_d_des, th_dd_des) -> tau`.
"""

from functools import partial
from typing import Callable

from jax import Array, jit
from jax import numpy as jnp


@jit
def ctrl_fb_pd(
    th: Array,
    th_d: Array,
    th_des: Array,
    th_d_des: Array,
    kp: Array = jnp.zeros((2, 2)),
    kd: Array = jnp.zeros((2, 2)),
) -> Array:
    """PD feedback on the error of the absolute link angles."""
    return kp @ (th_des - th) + kd @ (th_d_des - th_d)


@jit
def ctrl_fb_pd_rel(
    th: Array,
    th_d: Array,
    th_des: Array,
    th_d_des: Array,
    kp: Array = jnp.zeros((2, 2)),
    kd: Array = jnp.zeros((2, 2)),
) -> Array:
    """PD feedback on the error of the joint angles.

    The torque acts on the joints, so the error in the joint space gives a better
    result than the error in the link space.
    """
    th_rel = jnp.stack([th[0], th[1] - th[0]])
    th_d_rel = jnp.stack([th_d[0], th_d[1] - th_d[0]])
    th_des_rel = jnp.stack([th_des[0], th_des[1] - th_des[0]])
    th_d_des_rel = jnp.stack([th_d_des[0], th_d_des[1] - th_d_des[0]])

    return kp @ (th_des_rel - th_rel) + kd @ (th_d_des_rel - th_d_rel)


@partial(jit, static_argnums=(0,), static_argnames=("dynamical_matrices_fn",))
def ctrl_ff_gravity_compensation(
    dynamical_matrices_fn: Callable,
    th: Array,
    th_d: Array,
    th_des: Array,
    th_d_des: Array,
    th_dd_des: Array,
) -> Array:
    """Cancel the gravity torque at the measured state."""
    _, _, G = dynamical_matrices_fn(th, th_d)
    return G


@partial(jit, static_argnums=(0,), static_argnames=("dynamical_matrices_fn",))
def ctrl_ff_feedforward(
    dynamical_matrices_fn: Callable,
    th: Array,
    th_d: Array,
    th_des: Array,
    th_d_des: Array,
    th_dd_des: Array,
) -> Array:
    """Inverse dynamics at the desired state.

    The controller does not use the measured state, so it cannot correct an error.
    """
    M, C, G = dynamical_matrices_fn(th_des, th_d_des)
    return M @ th_dd_des + C @ th_d_des + G


@partial(jit, static_argnums=(0,), static_argnames=("dynamical_matrices_fn",))
def ctrl_ff_pd_plus(
    dynamical_matrices_fn: Callable,
    th: Array,
    th_d: Array,
    th_des: Array,
    th_d_des: Array,
    th_dd_des: Array,
) -> Array:
    """The PD+ (Paden-Panja) feedforward term.

    The matrices are evaluated at the measured state, and the desired speed and the
    desired acceleration give the torque. Together with a PD feedback term this makes
    the tracking error dynamics linear.
    """
    M, C, G = dynamical_matrices_fn(th, th_d)
    return M @ th_dd_des + C @ th_d_des + G
