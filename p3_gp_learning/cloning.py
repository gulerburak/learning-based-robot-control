"""Feedback controllers that a Gaussian process learned from a PD controller.

Behavioural cloning copies a controller from its data. The problem is that the GP sees
only the angles. It sees no speed and no reference, so it learned a torque that is
correct near the training path only. When the robot moves away from that path, the GP
applies a torque that belongs to a different point, and the error grows.

The uncertainty of the GP gives the solution. The variance is small where the training
data is, and it grows away from it. An extra torque

    tau_repel = -k_var * sigma * sign(d(sigma^2) / d(th))

moves the robot in the direction where the uncertainty decreases. That direction points
to the training path. The controller therefore gets a feedback to the path, although it
does not know the desired position.
"""

from typing import Callable

import numpy as np
import torch
from jax import Array
from jax import numpy as jnp

from jax_double_pendulum.utils import normalize_link_angles
from p3_gp_learning.wrappers import MultitaskGPRegressor


def link_to_relative_angles(th: Array) -> torch.Tensor:
    """Give the joint angles as a GP input of shape (1, 2)."""
    th_rel = normalize_link_angles(jnp.array([th[0], th[1] - th[0]]))
    return torch.tensor(np.array(th_rel)).unsqueeze(0).double()


def make_torque_cloning_controller(
    model: MultitaskGPRegressor, k_var: float = 0.0
) -> Callable:
    """Give a feedback controller that predicts the torque from the angles.

    Args:
        model: a GP that learned (th_1, th_2_rel) -> (tau_1, tau_2)
        k_var: the gain of the term that moves the robot away from uncertainty. With 0
            the controller is the pure cloned policy, and the closed loop diverges.
    """

    def ctrl_fb(th: Array, th_d: Array, th_des: Array, th_d_des: Array) -> Array:
        gp_input = link_to_relative_angles(th)

        with torch.no_grad():
            gp_output = model.predict(gp_input)
            tau_mean = gp_output.mean.flatten()
            tau_sigma = gp_output.stddev.flatten()

        if k_var == 0.0:
            return tau_mean.numpy()

        # Row 0 of the Jacobian gives the change of the first variance against both
        # angles. It is the direction in which the uncertainty grows.
        grad_var = model.variance_jacobian(gp_input)[0]
        tau_repel = -k_var * tau_sigma * np.sign(grad_var)

        return (tau_mean + tau_repel).squeeze().numpy()

    return ctrl_fb


def make_reference_cloning_controller(
    model: MultitaskGPRegressor,
    kp: float = 500.0,
    kd: float = 0.0,
    k_var: float = 0.0,
) -> Callable:
    """Give a feedback controller that predicts the next step of the path.

    The GP learned (th_1, th_2_rel) -> (delta th_1, delta th_2). The delta is the change
    of the angle over one time step, so it points along the path. A proportional gain
    makes a torque from it.

    Args:
        kp: the gain from the angle change to the torque [Nm/rad]
        kd: the damping gain [Nm s/rad]. The GP gives no speed feedback, so without
            damping the robot oscillates.
        k_var: the gain of the term that moves the robot away from uncertainty.
    """
    Kp = kp * np.eye(2)
    Kd = kd * np.eye(2)

    def ctrl_fb(th: Array, th_d: Array, th_des: Array, th_d_des: Array) -> Array:
        gp_input = link_to_relative_angles(th)

        with torch.no_grad():
            gp_output = model.predict(gp_input)
            delta_mean = gp_output.mean.numpy().flatten()
            delta_sigma = gp_output.stddev.numpy().flatten()

        delta = delta_mean
        if k_var != 0.0:
            # The diagonal of the Jacobian gives the change of each variance against its
            # own angle.
            grad_var = np.diag(model.variance_jacobian(gp_input).numpy())
            delta = delta + k_var * (-delta_sigma * np.sign(grad_var))

        return Kp @ delta + Kd @ (-np.asarray(th_d))

    return ctrl_fb
