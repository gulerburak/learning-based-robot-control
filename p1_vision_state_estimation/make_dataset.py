"""Make the image dataset.

There are two methods:

  --zip PATH   Unpack the dataset archive of the course.
  --render     Render 3600 images with the gymnasium pendulum environment.

Both methods write the files `image{i}.npz` and `label{i}.npz` into the output folder.
The image is a 500x500x3 array of `uint8`. The label is the link angle in rad, in the
range [0, 2*pi). The angle is zero when the link points up, and it increases
counterclockwise.
"""

import argparse
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from tqdm import tqdm

DEFAULT_DIR = Path("data") / "pendulum_images"
NUM_SAMPLES = 3600


def extract_zip(zip_path: Path, output_dir: Path):
    """Unpack the archive into one flat folder.

    The archive of the course holds a top folder. The files are written without it, so
    that the result is the same as the result of `render_dataset`.
    """
    if output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with ZipFile(zip_path, "r") as archive:
        members = [m for m in archive.namelist() if m.endswith(".npz")]
        for member in tqdm(members, desc="unpack"):
            with archive.open(member) as source:
                (output_dir / Path(member).name).write_bytes(source.read())

    print(f"Unpacked {len(members)} files into {output_dir}.")


def render_dataset(output_dir: Path, num_samples: int = NUM_SAMPLES):
    """Render one image for each angle, at equal steps over the full circle.

    The angles go from 0 to 2*pi, which is the range that the dataset of the course uses.
    """
    import gymnasium as gym

    output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset(seed=0)

    angles = np.linspace(0.0, 2 * np.pi, num_samples, endpoint=False)
    for index, angle in enumerate(tqdm(angles, desc="render")):
        env.unwrapped.state = np.array([angle, 0.0])
        image = env.render()
        np.savez_compressed(output_dir / f"image{index}.npz", np.asarray(image))
        np.savez_compressed(output_dir / f"label{index}.npz", np.asarray(angle))

    env.close()
    print(f"Wrote {num_samples} samples into {output_dir}.")


def make_gif(data_dir: Path, filepath: Path, step: int = 100):
    """Write a GIF that shows every `step`-th image of the dataset."""
    import imageio

    frames = [
        np.load(data_dir / f"image{index}.npz")["arr_0"]
        for index in range(0, NUM_SAMPLES, step)
    ]
    imageio.mimsave(filepath, frames, duration=0.1)
    print(f"Wrote {filepath}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, help="path to the dataset archive")
    parser.add_argument("--render", action="store_true", help="render with gymnasium")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--gif", action="store_true", help="also write a preview GIF")
    args = parser.parse_args()

    if args.zip:
        extract_zip(args.zip, args.output_dir)
    elif args.render:
        render_dataset(args.output_dir, args.num_samples)
    else:
        parser.error("Give --zip PATH or --render.")

    if args.gif:
        make_gif(args.output_dir, args.output_dir.parent / "pendulum_dataset.gif")


if __name__ == "__main__":
    main()
