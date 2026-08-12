"""A Lagrangian Neural Network for the 2-link robot.

The network does not learn the acceleration directly. It learns the two parts of the
Lagrangian L = T - U:

  * `MassMatrixNN` gives the mass matrix M(th). The network gives the lower triangle of
    a matrix L, and M = L @ L.T. This construction makes M symmetric and positive
    definite, which the physics requires.
  * `PotentialEnergyNN` gives the potential energy U(th).

The kinetic energy is T = 0.5 * th_d.T @ M(th) @ th_d. Automatic differentiation of the
Lagrangian then gives the matrices of the equation of motion:

    M @ th_dd + C @ th_d + G = tau

The model therefore obeys the conservation of energy by construction. A network that
learns the acceleration directly does not.
"""

from typing import Dict, Tuple

import jax
import numpy as np
from flax import linen as nn
from jax import Array, jit
from jax import numpy as jnp

from jax_double_pendulum.integrators import rk4_step
from jax_double_pendulum.utils import normalize_link_angles

NUM_HIDDEN = 32
NUM_LAYERS = 4


class MassMatrixNN(nn.Module):
    """Give a symmetric, positive definite mass matrix M(th)."""

    num_hidden: int = NUM_HIDDEN

    diagonal_shift = 0.001
    diagonal_eps = 0.002

    @nn.compact
    def __call__(self, th: Array) -> Array:
        num_dof = th.shape[-1]
        num_nn_outputs = int((num_dof**2 + num_dof) / 2)

        x = th
        for _ in range(NUM_LAYERS):
            x = nn.softplus(nn.Dense(self.num_hidden)(x))
        m = nn.Dense(num_nn_outputs)(x)

        l_diagonal, l_off_diagonal = jnp.split(m, np.array([num_dof]), axis=-1)

        # A positive diagonal keeps the matrix positive definite and invertible.
        l_diagonal = nn.softplus(l_diagonal + self.diagonal_shift) + self.diagonal_eps

        indices_diag = np.arange(num_dof, dtype=int) + 1
        indices_diag = (indices_diag * (indices_diag + 1) / 2 - 1).astype(int)
        indices_off_diag = np.setdiff1d(np.arange(num_nn_outputs), indices_diag)
        indices_nn_output = np.hstack((indices_diag, indices_off_diag))

        vec_tril = jnp.concatenate([l_diagonal, l_off_diagonal], axis=-1)[
            ..., indices_nn_output
        ]
        tril_mat = jnp.zeros((num_dof, num_dof))
        tril_mat = tril_mat.at[np.tril_indices(num_dof)].set(vec_tril[:])

        return tril_mat @ tril_mat.transpose()


class PotentialEnergyNN(nn.Module):
    """Give the potential energy U(th) as a scalar."""

    num_hidden: int = NUM_HIDDEN

    @nn.compact
    def __call__(self, th: Array) -> Array:
        x = th
        for _ in range(NUM_LAYERS):
            x = nn.softplus(nn.Dense(self.num_hidden)(x))
        return jnp.squeeze(nn.Dense(1)(x))


@jit
def kinetic_energy_fn(mass_matrix_nn_params: Dict, th: Array, th_d: Array) -> Array:
    M = MassMatrixNN().apply({"params": mass_matrix_nn_params}, th)
    return 0.5 * th_d @ M @ th_d


@jit
def potential_energy_fn(potential_energy_nn_params: Dict, th: Array) -> Array:
    return PotentialEnergyNN().apply({"params": potential_energy_nn_params}, th)


@jit
def lagrangian_fn(
    mass_matrix_nn_params: Dict,
    potential_energy_nn_params: Dict,
    th: Array,
    th_d: Array,
) -> Array:
    T = kinetic_energy_fn(mass_matrix_nn_params, th, th_d)
    U = potential_energy_fn(potential_energy_nn_params, th)
    return jnp.squeeze(T - U)


@jit
def mass_matrix_fn(mass_matrix_nn_params: Dict, th: Array) -> Array:
    return MassMatrixNN().apply({"params": mass_matrix_nn_params}, th)


@jit
def dynamical_matrices(
    mass_matrix_nn_params: Dict,
    potential_energy_nn_params: Dict,
    th: Array,
    th_d: Array,
) -> Tuple[Array, Array, Array]:
    """Give M, C and G from the derivatives of the Lagrangian.

    M is the Hessian of L against the speed. C comes from the Christoffel symbols of the
    first kind, which need the derivative of M against the angle. G is the gradient of
    the potential energy.

    Returns:
        M: shape (2, 2), C: shape (2, 2), G: shape (2,)
    """
    th_normalized = normalize_link_angles(th)

    M = jax.hessian(lagrangian_fn, argnums=3)(
        mass_matrix_nn_params, potential_energy_nn_params, th_normalized, th_d
    )

    dM_dth = jax.jacobian(jax.hessian(lagrangian_fn, argnums=3), argnums=2)(
        mass_matrix_nn_params, potential_energy_nn_params, th_normalized, th_d
    )

    C = (
        0.5
        * (dM_dth + jnp.transpose(dM_dth, (0, 2, 1)) - jnp.transpose(dM_dth, (2, 0, 1)))
        @ th_d
    )

    G = jax.grad(potential_energy_fn, argnums=1)(
        potential_energy_nn_params, th_normalized
    )

    return M, C, G


@jit
def continuous_forward_dynamics(
    mass_matrix_nn_params: Dict,
    potential_energy_nn_params: Dict,
    th: Array,
    th_d: Array,
    tau: Array = jnp.zeros((2,)),
) -> Array:
    """Give the angular acceleration for a torque."""
    M, C, G = dynamical_matrices(
        mass_matrix_nn_params, potential_energy_nn_params, th, th_d
    )
    return jnp.linalg.inv(M) @ (tau - C @ th_d - G)


def continuous_state_space_dynamics(
    mass_matrix_nn_params: Dict,
    potential_energy_nn_params: Dict,
    x: Array,
    tau: Array,
) -> Tuple[Array, Array]:
    """Give the time derivative of the state x = [th, th_d] and the output y = th."""
    th, th_d = jnp.split(x, 2)
    th_dd = continuous_forward_dynamics(
        mass_matrix_nn_params, potential_energy_nn_params, th, th_d, tau
    )
    return jnp.concatenate([th_d, th_dd]), th


@jit
def discrete_forward_dynamics(
    mass_matrix_nn_params: Dict,
    potential_energy_nn_params: Dict,
    dt: Array,
    th_curr: Array,
    th_d_curr: Array,
    tau: Array = jnp.zeros((2,)),
) -> Tuple[Array, Array, Array]:
    """Make one step of the learned dynamics with the RK4 integrator."""
    th_dd = continuous_forward_dynamics(
        mass_matrix_nn_params, potential_energy_nn_params, th_curr, th_d_curr, tau
    )

    x_next = rk4_step(
        ode_fun=lambda _x: continuous_state_space_dynamics(
            mass_matrix_nn_params, potential_energy_nn_params, _x, tau
        )[0],
        x=jnp.concatenate([th_curr, th_d_curr]),
        dt=dt,
    )
    th_next, th_d_next = jnp.split(x_next, 2)

    return th_next, th_d_next, th_dd
