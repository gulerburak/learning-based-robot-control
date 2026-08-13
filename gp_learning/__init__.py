"""Gaussian processes and behavioural cloning for robot control.

The package shows two things that a Gaussian process gives and a neural network does not:

  * The model says how sure it is. The variance grows away from the training data.
  * That variance is useful in the control loop. A torque term that moves the robot to a
    low-variance state makes an unstable cloned policy stable.
"""

import os
import warnings

import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")
