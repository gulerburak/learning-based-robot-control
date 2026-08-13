"""Gaussian process models.

Three model types are here:

  * `ExactGPModel` — the exact solution. The cost grows with the cube of the number of
    samples, so it works for some hundreds of samples only.
  * `SVGPModel` — a sparse variational model with one output. A small set of inducing
    points holds the information of the full dataset, so the cost stays low.
  * `MultitaskGPModel` — a sparse variational model with more than one output. Each
    output has its own kernel parameters, and the outputs are independent.

The kernel makes the assumption about the function:
  * RBF gives a smooth function.
  * Matern 5/2 gives a function that is less smooth. That fits a mechanical system
    better.
  * Periodic gives a function that repeats. A joint angle is periodic, so the model then
    also works for an angle that it never saw.
  * ARD gives one lengthscale for each input. A large lengthscale means that the input
    has a small effect.
"""

from typing import Callable

import gpytorch
import torch

TWO_PI = 2 * torch.pi
PERIOD_TOLERANCE = 0.01


# The `batch_shape` goes to the ScaleKernel only. The outputs then have their own
# output scale, but they share the lengthscales. This keeps the number of parameters
# small.


def rbf_kernel(num_dims: int = None, batch_shape: torch.Size = torch.Size([])):
    """A smooth kernel. With `num_dims` it gets one lengthscale for each input."""
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=num_dims), batch_shape=batch_shape
    )


def matern_kernel(num_dims: int = None, batch_shape: torch.Size = torch.Size([])):
    """A Matern 5/2 kernel with ARD."""
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=num_dims),
        batch_shape=batch_shape,
    )


def periodic_kernel(num_dims: int = None, batch_shape: torch.Size = torch.Size([])):
    """A periodic kernel with the period held at 2*pi.

    The constraint is a narrow interval and not a fixed value, so the optimizer can still
    move the parameter. The learned period must come out near 2*pi.
    """
    constraint = gpytorch.constraints.Interval(
        lower_bound=torch.tensor([TWO_PI - PERIOD_TOLERANCE]),
        upper_bound=torch.tensor([TWO_PI + PERIOD_TOLERANCE]),
        initial_value=torch.tensor([TWO_PI]),
    )
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.PeriodicKernel(period_length_constraint=constraint),
        batch_shape=batch_shape,
    )


class ExactGPModel(gpytorch.models.ExactGP):
    """An exact GP with a zero mean and an RBF kernel."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = rbf_kernel()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class SVGPModel(gpytorch.models.ApproximateGP):
    """A sparse variational GP with one output.

    Args:
        inducing_points: shape (num_inducing, num_features). The positions are learned.
        constant_mean: True gives a mean that the model learns. Use it when the data is
            not near zero. False gives a zero mean, which is safer far from the data.
        ard: True gives one lengthscale for each input.
    """

    def __init__(self, inducing_points, constant_mean: bool = False, ard: bool = False):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = (
            gpytorch.means.ConstantMean() if constant_mean else gpytorch.means.ZeroMean()
        )
        num_dims = inducing_points.size(-1) if ard and inducing_points.dim() > 1 else None
        self.covar_module = rbf_kernel(num_dims=num_dims)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    """A sparse variational GP with several independent outputs.

    Args:
        num_tasks: the number of outputs.
        inducing_points: shape (num_inducing, num_features).
        kernel_fn: `matern_kernel`, `periodic_kernel` or `rbf_kernel`.
        constant_mean: see `SVGPModel`.
    """

    def __init__(
        self,
        num_tasks: int,
        inducing_points,
        kernel_fn: Callable = matern_kernel,
        constant_mean: bool = False,
    ):
        batch_shape = torch.Size([num_tasks])

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            task_dim=-1,
        )
        super().__init__(variational_strategy)

        self.mean_module = (
            gpytorch.means.ConstantMean(batch_shape=batch_shape)
            if constant_mean
            else gpytorch.means.ZeroMean(batch_shape=batch_shape)
        )
        self.covar_module = kernel_fn(
            num_dims=inducing_points.size(-1), batch_shape=batch_shape
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )
