"""The two convolutional networks.

The two networks have the same body. Only the width of the output layer changes.
"""

import torch
from torch import nn


class PendulumCNN(nn.Module):
    """A small CNN that maps a 1x24x24 image to `num_outputs` values.

    The image size and the two pooling layers fix the size of the flat vector:
    24 -> conv(3) -> 22 -> pool(2) -> 11 -> conv(3) -> 9 -> pool(2) -> 4.
    """

    def __init__(self, num_outputs: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.fc1 = nn.Linear(10 * 4 * 4, 30)
        self.fc2 = nn.Linear(30, num_outputs)

        self.act_fn = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act_fn(self.conv1(x)))
        x = self.pool2(self.act_fn(self.conv2(x)))
        x = self.flatten(x)
        x = self.act_fn(self.fc1(x))
        return self.fc2(x)


class CNNTheta(PendulumCNN):
    """Direct model. The output is the link angle in rad."""

    def __init__(self):
        super().__init__(num_outputs=1)


class CNNTrig(PendulumCNN):
    """Indirect model. The outputs are the sine and the cosine of the link angle."""

    def __init__(self):
        super().__init__(num_outputs=2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
