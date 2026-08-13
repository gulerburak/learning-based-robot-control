# Results

Every number here comes from a run of the scripts in this repo.

The robot is a planar 2-link pendulum. `ROBOT_PARAMS = {l1: 2, lc1: 1, m1: 10, j1: 3,
l2: 1, lc2: 0.5, m2: 6, j2: 2, g: 9.81}`. The reference is an ellipse in the operational
space: `ELLIPSE_PARAMS = {omega: 72 deg/s, rx: 1.75, ry: 1.25, ell_angle: 45 deg,
x0: 0.4, y0: 0.4}`. The simulation runs for 10 s at `dt = 1e-2`, so N = 1000.

`RMSE x` is the norm of the operational-space root-mean-square error, in metres. It is
the main quality measure for all control tasks.

---

## Project 1 — vision-based state estimation

Dataset: 3600 images, 500×500×3, `uint8`. Split: 50 % train, 20 % validation, 30 % test
(1800 / 720 / 1080). Batch size 32. 50 epochs. 3 runs, with seeds 0, 1 and 2.

| Model | Parameters | Optimizer | Test error, mean ± sd |
|---|---|---|---|
| `CNNTheta` (direct angle) | 8071 | SGD, lr 1e-3 | 0.5075 ± 0.0857 rad |
| `CNNTrig` (sin and cos) | 8102 | SGD, lr 1e-2 | 0.0126 ± 0.0033 rad |

Per run:

| Run | Theta: test loss [rad²] | Theta: error [rad] | Trig: test loss | Trig: error [rad] |
|---|---|---|---|---|
| 0 | 0.3902 | 0.4216 | 0.0001769 | 0.008448 |
| 1 | 0.6067 | 0.6245 | 0.0003716 | 0.012800 |
| 2 | 0.4368 | 0.4765 | 0.0004233 | 0.016610 |

The loss magnitudes differ because the outputs have different ranges: the angle has a
range of 2π, and the sine and the cosine have a range of 2. Only the error in rad is
comparable between the two models.

The error must wrap at the full circle. The labels of the dataset go from 0 to 2π, and
`atan2` gives a value in [−π, π]. A direct difference of the two therefore counts a full
circle for a sample at the end of the range, although the two angles are almost the same.
`train.angular_error` sends the difference through the sine and the cosine.

---

## Project 2 — model-based control, LNN, ILC

### Classical controllers

| Controller | Gains | RMSE x [m] |
|---|---|---|
| PD on link angles | kp = 5000·I, kd = 500·I | 0.0635 |
| PD on joint angles | kp = 5000·I, kd = 500·I | 0.0501 |
| PD + gravity compensation | kp = 5000·I, kd = 500·I | 0.0269 |
| PD + feedforward | kp = 5000·I, kd = 50·I | 0.0619 |
| PD+ (Paden–Panja) | kp = 500·I, kd = 50·I | 0.0491 |

The initial state has an error of [0.1, 0.2] rad against the first point of the path, so
the feedback term must do work.

### Lagrangian Neural Network

Dataset: 250 rollouts, 10 s each, `dt = 1e-2`. Random start angles in [−π, π], random
start speeds in [−2π, 2π] rad/s, random torques in [−100, 100] N·m.
Total: 250 × 999 = 249 750 samples. Split 80/20 → 199 800 train, 49 950 validation.

Training: 250 epochs, batch 250, AdamW, base learning rate 7e-4, 10 warmup epochs, cosine
decay, weight decay 0.

| Measure | Value |
|---|---|
| Best validation loss | 2.4391e-07, at epoch 248 |
| RMSE of the next angular speed | 4.9387e-04 rad/s |
| Free rollout against the true model, RMSE x | 0.0392 m |
| PD+ with the learned M, C, G, RMSE x | 0.0493 m |

A short run does not give this result. With 60 rollouts and 60 epochs the validation loss
stops at 1.4e-02, and the free rollout leaves the true motion after some seconds
(3.76 m). The full training needs approximately one hour on a CPU.

Reference values of the untrained network with seed 0, `th = [0, 0]`, `th_d = [π, π]`,
`tau = [1, 1]`, `dt = 1e-2`. `tests/test_lnn.py` checks them:

```
M      = [[ 0.54595144, -0.53372961], [-0.53372961,  0.97476694]]
U      = -0.3061547720368191
L      =  2.542899071686838
C      = [[-0.00735404,  0.17485702], [ 0.09947973, -0.12300335]]
G      = [ 0.03959358, -0.05480955]
th_dd  = [ 4.14726048,  3.42874464]
th_next   = [0.03162163, 0.0315865 ]
th_d_next = [3.18256705, 3.17561977]
```

### Linearization and ILC

The controller uses a wrong model, and the simulation uses the true model. The masses and
the inertias of the wrong model are multiplied by a perturbation factor.

| Method | Setup | RMSE x [m] |
|---|---|---|
| PD-ILC | 500 iterations, kp_ilc = 2e-5, kd_ilc = 2e-3, factor 3 | 0.0586 |
| Q-ILC | 1000 iterations, Q = 1e0·I, S = 5e-4·I, factor 1.8 | 0.0335 |

The feedback gains are fixed at kp = 500·I and kd = 50·I. The lifted-system matrix `P`
and the gain matrix `L_opt` have a size of 1998 × 1998.

---

## Project 3 — Gaussian processes and behavioural cloning

### One input, tensile test data

| Model | Kernel | Training | Lengthscale | Output scale | Noise |
|---|---|---|---|---|---|
| Exact GP | Zero mean, Scale(RBF) | Adam 0.1, 500 epochs | 0.8882 | 9.7062 (raw) | 0.0009 |
| SVGP, 10 inducing points | Zero mean, Scale(RBF) | Adam 0.01, 10000 epochs | 2.3553 | 8.7074 (raw) | 0.0303 |

The sparse model has a larger lengthscale and more noise. With 10 inducing points it
cannot follow the sharp yield point of the steel sample.

### Eight inputs, concrete strength data

1030 samples, 80/20 split, 300 inducing points, ARD RBF kernel, 2000 epochs.

| Mean function | MAE train | MAE test | Calibration train | Calibration test | Unsafe train | Unsafe test |
|---|---|---|---|---|---|---|
| Constant | 3.2007 | 5.2606 | 0.9320 | 0.8010 | 0.0182 | 0.0340 |
| Zero | 3.2923 | 5.8363 | 0.9211 | 0.7718 | 0.0182 | 0.0437 |

"Calibration" is the fraction of true values inside the 2σ interval. "Unsafe" is the
fraction of samples where the lower 2σ bound is above the true strength. Such a prediction
says that the concrete is stronger than it is.

ARD lengthscales, constant mean:
`[61.11, 68.58, 51.46, 44.97, 32.71, 61.08, 65.17, 30.90]`, output scale 32.91,
constant 21.39. The eighth input is the age of the sample. Its small lengthscale shows
that it has a large effect.

ARD lengthscales, zero mean:
`[62.58, 70.23, 54.48, 49.18, 35.56, 62.49, 67.08, 42.50]`, output scale 39.02.

The unsafe share is a small count on 206 test samples, and the inducing points start at
random positions. The value moves between runs. Do not read one run as proof of which
mean function is safer.

### Forward dynamics with a multi-output GP

Matérn 5/2 with ARD, 6 inputs, 2 outputs, 100 inducing points, 30 epochs, Adam 0.01.

Large-oscillation dataset: lengthscales `[7.7341, 7.1021, 0.7993, 0.8621, 0.6931,
0.6931]`, output scale `[2.2956, 2.6907]`, ELBO −16.2937.

The first two inputs are the angular speeds. Their large lengthscales show that the
acceleration changes slowly with the speed. The angle inputs have short lengthscales,
because gravity changes quickly with the angle.

### Forward dynamics with an MLP

One hidden layer of 1000 units, 200 epochs, Adam. The final training RMSE is
1.8721 rad/s² on the large-oscillation data and 0.0289 rad/s² on the small-oscillation
data.

The large-oscillation data is more difficult, because it covers the full state space and
holds much larger accelerations. A small error on that data is therefore not the same as a
small error on the other data. The comparison that counts is the phase portrait, not this
number.

### Behavioural cloning

The learned period of the periodic kernel is 6.2760 rad for the torque model and
6.2853 rad for the delta-angle model. The true value is 2π = 6.2832, and the constraint
interval is [2π − 0.01, 2π + 0.01].

| Controller | Tuned values | RMSE of the tip | Largest error |
|---|---|---|---|
| Clone of the feedback torque | — | 0.996 m | 2.08 m |
| The same, plus variance repulsion | k_var = 2.0 | 0.157 m | 0.40 m |
| Clone of the delta angle | kp = 500 | 2.558 m | 4.62 m |
| The same, plus damping and repulsion | kp = 500, kd = 2.0, k_var = 1.0 | 2.025 m | 3.30 m |
