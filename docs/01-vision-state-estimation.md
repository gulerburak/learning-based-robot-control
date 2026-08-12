# Project 1 — Vision-based state estimation

**Code:** `p1_vision_state_estimation/` · **Library:** PyTorch

## The problem

A controller needs the state of the robot. A camera gives an image, not a state. This
project trains a small convolutional network that reads the angle of a pendulum from an
image.

The dataset holds 3600 images of 500x500 pixels. Each image has one label: the angle of
the link in rad, in the range [−π, π]. The angle is zero when the link points up.

## The method

The image pipeline makes the problem small:

```
500x500 RGB  ->  grayscale  ->  centre crop 240  ->  resize 24x24  ->  1x24x24 float
```

The network is small on purpose: two convolution layers, two average-pool layers and two
linear layers. It has approximately 8000 parameters.

The interesting part is the output. The same body gets two different heads:

| Model | Output | Loss against | Angle from |
|---|---|---|---|
| `CNNTheta` | one value | the angle | the output |
| `CNNTrig` | two values | the sine and the cosine | `atan2(sin, cos)` |

## The result

Both models train for 50 epochs with SGD, in three runs with different seeds.

| Model | Parameters | Mean absolute error |
|---|---|---|
| `CNNTheta`, direct | 8071 | **0.5301 ± 0.2476 rad** (about 30 degrees) |
| `CNNTrig`, sine and cosine | 8102 | **0.0706 ± 0.0322 rad** (about 4 degrees) |

31 more parameters make the model 7.5 times more accurate.

## Why the difference is so large

The angle wraps at ±π. Two angles that are near to each other, for example 3.13 rad and
−3.13 rad, have a difference of 0.02 rad. The loss function of the direct model sees a
difference of 6.26 rad and gives a very large penalty. The network must therefore learn a
function with a step in it, and a step is difficult for a smooth network.

The sine and the cosine have no step. They are continuous functions of the angle, so the
network learns them easily. `atan2` then gives the angle back, and it gives the correct
quadrant.

The sine alone is not sufficient, because sin(θ) = sin(π − θ). Two different angles then
give the same output.

The figure `outputs/p1/error_vs_angle.pdf` shows the effect directly. The error of the
direct model is largest near ±π.

## How to run

```bash
# Make the images (needs gymnasium), or unpack an archive that you have
python -m p1_vision_state_estimation.make_dataset --render

# Train both models
python -m p1_vision_state_estimation.run --model both --epochs 50 --runs 3
```

The script writes the loss curves and the error-against-angle figure into `outputs/p1/`.

## Changes against the original notebook

1. The original evaluated the trig model but left it in training mode, because of a
   copy-and-paste fault (`model_theta.eval()`). The port corrects this.
2. The original made the train/test split before it set the seed, so the split was not
   repeatable. The port sets the seed first.
3. The original made a validation set and never used it. The port measures the validation
   error in each epoch.
4. The original printed the loss only. The port also makes figures.
