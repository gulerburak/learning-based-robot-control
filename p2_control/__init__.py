"""Model-based control, learned dynamics and iterative learning control.

The package works on a planar 2-link robot. `jax_double_pendulum` gives the simulator.

The JAX settings must run before any array is made. The ILC tasks need 64-bit floats,
because they invert matrices with a size of almost 2000 x 2000.
"""

import os
import warnings

import jax
import matplotlib

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

# The plot helpers of `jax_double_pendulum` call `plt.show()`. A file backend keeps the
# scripts non-interactive. Set MPLBACKEND to see the figures on the screen.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")
