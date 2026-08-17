"""The image dataset and the data loaders."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

IMAGE_SIZE = 24
CROP_SIZE = 240


def build_transform() -> transforms.Compose:
    """Make the image pipeline.

    The centre crop removes the border of the rendered scene. The small output keeps the
    network small, and the pendulum stays visible.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.CenterCrop(CROP_SIZE),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        ]
    )


class PendulumImageDataset(Dataset):
    """Images of a pendulum and the angle of the link.

    Each sample is a tuple `(image, theta, trig)`:
      image: float tensor of shape (1, 24, 24), values in [0, 1]
      theta: float tensor of shape (1,), the link angle in rad, in [0, 2*pi)
      trig:  float tensor of shape (2,), the sine and the cosine of the angle

    Note the range of `theta`. It is [0, 2*pi), and `atan2` gives a value in [-pi, pi].
    Any error measure must therefore use the wrap of the full circle. See
    `train.angular_error`.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"{self.data_dir} does not exist. Run "
                "`python -m vision_state_estimation.make_dataset` first."
            )
        self.transform = build_transform()
        self.num_samples = self._count_samples()
        if self.num_samples == 0:
            raise FileNotFoundError(f"{self.data_dir} holds no image*.npz file.")

    def _count_samples(self) -> int:
        index = 0
        while (self.data_dir / f"image{index}.npz").is_file():
            index += 1
        return index

    def __len__(self) -> int:
        return self.num_samples

    def angle(self, index: int) -> float:
        """Give the label of one sample, without the image."""
        return float(np.load(self.data_dir / f"label{index}.npz")["arr_0"])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = np.load(self.data_dir / f"image{index}.npz")["arr_0"]
        theta = self.angle(index)

        x = self.transform(image)
        trig = torch.FloatTensor([np.sin(theta), np.cos(theta)])
        return x, torch.FloatTensor([theta]), trig


def load_dataloaders(
    data_dir: Path,
    val_ratio: float = 0.2,
    test_ratio: float = 0.3,
    batch_size: int = 32,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Split the dataset and make the three loaders.

    The seed controls the split, so the same seed always gives the same three subsets.
    """
    dataset = PendulumImageDataset(data_dir)

    num_val = int(val_ratio * len(dataset))
    num_test = int(test_ratio * len(dataset))
    num_train = len(dataset) - num_val - num_test

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [num_train, num_val, num_test], generator=generator
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size),
    )
