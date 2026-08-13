"""Check the Lagrangian Neural Network against known reference values.

The values come from an untrained network with seed 0. They test the construction of the
mass matrix and the derivatives of the Lagrangian, not the quality of a trained model.

    python -m pytest tests/test_lnn.py
"""

import jax
from jax import numpy as jnp
from jax import random

from robot_control.lnn import (
    MassMatrixNN,
    PotentialEnergyNN,
    continuous_forward_dynamics,
    discrete_forward_dynamics,
    dynamical_matrices,
    lagrangian_fn,
)

RTOL, ATOL = 1e-4, 1e-7

TH = jnp.zeros((2,))
TH_D = jnp.pi * jnp.ones((2,))
TAU = jnp.ones((2,))
DT = 1e-2


def make_params():
    rng = random.PRNGKey(seed=0)
    mass_matrix_nn_params = MassMatrixNN().init(rng, jnp.ones((2,)))["params"]
    potential_energy_nn_params = PotentialEnergyNN().init(rng, jnp.ones((2,)))["params"]
    return mass_matrix_nn_params, potential_energy_nn_params


def test_mass_matrix():
    mass_matrix_nn_params, _ = make_params()
    M = MassMatrixNN().apply({"params": mass_matrix_nn_params}, TH)

    target = jnp.array([[0.54595144, -0.53372961], [-0.53372961, 0.97476694]])
    assert jnp.allclose(M, target, rtol=RTOL, atol=ATOL)
    assert jnp.allclose(M, M.T), "The mass matrix must be symmetric."
    assert jnp.all(jnp.linalg.eigvalsh(M) > 0), "The mass matrix must be positive."


def test_potential_energy():
    _, potential_energy_nn_params = make_params()
    U = PotentialEnergyNN().apply({"params": potential_energy_nn_params}, TH)

    assert U.shape == (), "The potential energy must be a scalar."
    assert jnp.allclose(U, jnp.array(-0.3061547720368191), rtol=RTOL, atol=ATOL)


def test_lagrangian():
    params = make_params()
    L = lagrangian_fn(*params, TH, TH_D)
    assert jnp.allclose(L, jnp.array(2.542899071686838), rtol=RTOL, atol=ATOL)


def test_dynamical_matrices():
    params = make_params()
    M, C, G = dynamical_matrices(*params, TH, TH_D)

    assert jnp.allclose(
        C,
        jnp.array([[-0.00735404, 0.17485702], [0.09947973, -0.12300335]]),
        rtol=RTOL,
        atol=ATOL,
    )
    assert jnp.allclose(
        G, jnp.array([0.03959358, -0.05480955]), rtol=RTOL, atol=ATOL
    )


def test_forward_dynamics():
    params = make_params()
    th_dd = continuous_forward_dynamics(*params, TH, TH_D, TAU)
    assert jnp.allclose(
        th_dd, jnp.array([4.14726048, 3.42874464]), rtol=RTOL, atol=ATOL
    )


def test_discrete_forward_dynamics():
    params = make_params()
    th_next, th_d_next, _ = discrete_forward_dynamics(*params, DT, TH, TH_D, TAU)

    assert jnp.allclose(
        th_next, jnp.array([0.03162163, 0.0315865]), rtol=RTOL, atol=ATOL
    )
    assert jnp.allclose(
        th_d_next, jnp.array([3.18256705, 3.17561977]), rtol=RTOL, atol=ATOL
    )
