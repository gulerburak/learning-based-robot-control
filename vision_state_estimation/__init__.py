"""Estimation of a pendulum angle from an image, with a small CNN.

The package compares two output representations. The direct model predicts the angle. The
indirect model predicts the sine and the cosine of the angle, and `atan2` gives the angle.
"""

from vision_state_estimation.models import CNNTheta, CNNTrig, PendulumCNN

__all__ = ["CNNTheta", "CNNTrig", "PendulumCNN"]
