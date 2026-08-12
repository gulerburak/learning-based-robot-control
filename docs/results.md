# Recorded results

These numbers come from the stored outputs of the original notebooks. They are the
targets for the port. A run of the code in this repo must agree with them.

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
| `CNNTheta` (direct angle) | 8071 | SGD, lr 1e-3 | **0.5301 ± 0.2476 rad** |
| `CNNTrig` (sin and cos) | 8102 | SGD, lr 1e-2 | **0.0706 ± 0.0322 rad** |

Per run:

| Run | Theta: train loss, first → last | Theta: test loss [rad²] | Theta: MAE [rad] |
|---|---|---|---|
| 0 | 8.152 → 0.4016 | 0.8850 | 0.8272 |
| 1 | 12.437 → 0.3801 | 0.4831 | 0.5419 |
| 2 | 10.055 → 0.3529 | 0.3015 | 0.2211 |

| Run | Trig: train loss, first → last | Trig: test loss | Trig: MAE [rad] |
|---|---|---|---|
| 0 | 0.5016 → 0.000262 | 0.0002502 | 0.06633 |
| 1 | 0.5037 → 0.000745 | 0.0015900 | 0.11200 |
| 2 | 0.5010 → 0.000543 | 0.0003518 | 0.03347 |

The sin/cos model is 7.5 times more accurate. The loss magnitudes differ because the
outputs have different ranges: the angle has a range of 2π, and the sine and the cosine
have a range of 2.

---

## Project 2 — model-based control, LNN, ILC

### Task 2a — classical controllers

| Task | Controller | Gains | RMSE x [m] | Limit |
|---|---|---|---|---|
| 2a.1.1 | PD on link angles | kp = 5000·I, kd = 500·I | **0.0635** | < 0.1 |
| 2a.1.2 | PD on joint angles | kp = 5000·I, kd = 500·I | **0.0501** | — |
| 2a.2 | PD + gravity compensation | kp = 5000·I, kd = 500·I | **0.0269** | < 0.06 |
| 2a.3 | PD + feedforward | kp = 5000·I, kd = 50·I | **0.0619** | — |
| 2a.4 | PD+ (Paden–Panja) | kp = 500·I, kd = 50·I | **0.0491** | < 0.06 |

The initial state has an error: `th_0 = traj_ts["th_ts"][0] - [0.1, 0.2]`.

Note: task 2a.2 asked for the smallest `kp` that holds the limit. The value stayed at
5000, the same as in 2a.1.

### Task 2c — Lagrangian Neural Network

Dataset: 250 rollouts, 10 s each, `dt = 1e-2`. Random start angles in [−π, π], random
start speeds in [−2π, 2π] rad/s, random torques in [−100, 100] N·m.
Total: 250 × 999 = **249 750 samples**. Split 80/20 → 199 800 train, 49 950 validation.

Training: 250 epochs, batch 250, AdamW, base learning rate 7e-4, 10 warmup epochs, cosine
decay, weight decay 0.

| Measure | Value |
|---|---|
| Best validation loss | **2.439e-07**, at epoch 247 |
| RMSE of the next angular speed | 4.94e-04 rad/s |
| 2c.4 rollout against the true model, RMSE x | **0.0392 m** (full marks) |
| 2c.5 PD+ with the learned M, C, G, RMSE x | **0.0493 m** (full marks) |

Reference values for the LNN unit test, with seed 0, `th = [0, 0]`, `th_d = [π, π]`,
`tau = [1, 1]`, `dt = 1e-2`:

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

### Task 2d — linearization and ILC

The controller uses a **wrong** model, and the simulation uses the true model. The masses
and the inertias of the wrong model are multiplied by a perturbation factor.

| Task | Setup | RMSE x [m] |
|---|---|---|
| 2d.2 PD-ILC | 500 iterations, kp_ilc = 2e-5, kd_ilc = 2e-3, factor 3 | **0.0586** (full marks) |
| 2d.3 Q-ILC | 1000 iterations, Q = 1e0·I, S = 5e-4·I, factor 1.8 | **0.0335** (limit 0.04) |

The feedback gains are fixed at kp = 500·I and kd = 50·I. The lifted-system matrix `P`
and the gain matrix `L_opt` have a size of 1998 × 1998.

---

## Project 3 — Gaussian processes and behavioural cloning

### Task 3a — one input, tensile test data

| Model | Kernel | Training | Lengthscale | Output scale | Noise |
|---|---|---|---|---|---|
| Exact GP | Zero mean, Scale(RBF) | Adam 0.1, 500 epochs | 0.8882 | 9.7062 (raw) | 0.0009 |
| SVGP, 10 inducing points | Zero mean, Scale(RBF) | Adam 0.01, 10000 epochs | 2.3553 | 8.7074 (raw) | 0.0303 |

The sparse model has a larger lengthscale and more noise. With 10 inducing points it
cannot follow the sharp yield point of the steel sample.

### Task 3b — eight inputs, concrete strength data

1030 samples, 80/20 split, 300 inducing points, ARD RBF kernel, 2000 epochs.

| Mean function | MAE train | MAE test | Calibration train | Calibration test | Unsafe train | Unsafe test |
|---|---|---|---|---|---|---|
| Constant | 3.3817 | **6.3547** | 0.9223 | 0.7573 | 0.0158 | 0.0874 |
| Zero | 3.4130 | 6.8418 | 0.9260 | 0.7816 | 0.0170 | **0.0583** |

"Calibration" is the fraction of true values inside the 2σ interval. "Unsafe" is the
fraction of samples where the lower 2σ bound is above the true strength. Such a
prediction says that the concrete is stronger than it is.

ARD lengthscales, constant mean:
`[63.04, 67.36, 60.31, 47.61, 33.12, 61.09, 65.37, 14.29]`, output scale 33.00,
constant 23.9963. The eighth input is the age of the sample. Its small lengthscale shows
that it has the largest effect.

ARD lengthscales, zero mean:
`[63.08, 68.72, 58.43, 50.26, 34.05, 61.25, 65.35, 40.03]`, output scale 38.99.

### Task 3d — forward dynamics with a multi-output GP

Matérn 5/2 with ARD, 6 inputs, 2 outputs, 100 inducing points, 30 epochs, Adam 0.01.

Large-oscillation dataset: lengthscales `[7.7341, 7.1021, 0.7993, 0.8621, 0.6931,
0.6931]`, output scale `[2.2956, 2.6907]`, ELBO −16.2937.

The first two inputs are the angular speeds. Their large lengthscales show that the
acceleration changes slowly with the speed. The angle inputs have short lengthscales,
because gravity changes quickly with the angle.

### Task 3f and 3g — behavioural cloning

| Task | Model | Tuned values | Result |
|---|---|---|---|
| 3f.5 | GP that clones the feedback torque | — | The closed loop diverges |
| 3f.6 | The same GP, plus variance repulsion | k_var = 2.0 | The closed loop is stable |
| 3g.4 | GP that clones the delta angle | kp = 500 | The robot leaves the path |
| 3g.6 | The same GP, plus repulsion and damping | kp = 500, kd = 2.0, k_var = 1.0 | The robot follows the path |

The learned period of the periodic kernel is 6.2760 rad. The true value is 2π = 6.2832.
The constraint interval was [2π − 0.01, 2π + 0.01].

---

# What the port reproduced

The table shows a run of the code in this repo against the recorded value above.

| Task | Recorded | This repo | Same? |
|---|---|---|---|
| 2a.1.1 PD, link angles | 0.0635 m | 0.0635 m | yes |
| 2a.1.2 PD, joint angles | 0.0501 m | 0.0501 m | yes |
| 2a.2 PD + gravity compensation | 0.0269 m | 0.0269 m | yes |
| 2a.3 PD + feedforward | 0.0619 m | 0.0619 m | yes |
| 2a.4 PD+ | 0.0491 m | 0.0491 m | yes |
| 2c.2 LNN reference values | see above | all 6 tests pass | yes |
| 2d.2 PD-ILC, 500 iterations | 0.0586 m | 0.0586 m | yes |
| 2d.3 Q-ILC, 1000 iterations | 0.0335 m | 0.0335 m | yes |
| 3a.1 exact GP lengthscale / noise | 0.8882 / 0.0009 | 0.8882 / 0.0009 | yes |
| 3a.2 sparse GP lengthscale / noise | 2.3553 / 0.0303 | 2.3553 / 0.0303 | yes |
| 3d ARD lengthscales, large oscillation | [7.7341, 7.1021, 0.7993, 0.8621, 0.6931, 0.6931] | the same | yes |
| 3d output scale, large oscillation | [2.2956, 2.6907] | the same | yes |
| 3f learned period of the kernel | 6.2760 rad | 6.2760 rad | yes |
| 3b MAE, constant / zero mean | 6.35 / 6.84 MPa | 5.26 / 5.84 MPa | no, see the note |
| 3b unsafe share, constant / zero | 0.087 / 0.058 | 0.034 / 0.044 | no, see the note |

**The note about task 3b.** The port chooses its 300 inducing points with
`torch.randperm`, and the original used `np.random.choice`. The two runs therefore start
at different points. The order of the error stays the same (the constant mean is better),
but the unsafe share is a small count on 206 test samples and it moves between runs. Do
not read one run as proof of which mean function is safer.

**Tasks that need a long run.** The results of tasks 2c.3 to 2c.5 need the full training:
250 simulations and 250 epochs, which is approximately one hour on a CPU. A short run (60
simulations, 60 epochs) gives a validation loss of 1.4e-02 and a rollout error of 3.76 m.
That shows the pipeline, not the result.

**Tasks 3f and 3g.** The original notebooks recorded no error value for these tasks, so
there is nothing to compare. A run of this repo gives:

| Task | Controller | RMSE of the tip | Largest error |
|---|---|---|---|
| 3f | the cloned policy alone | 0.996 m | 2.08 m |
| 3f | plus the variance term, k_var = 2.0 | 0.157 m | 0.40 m |
| 3g | the cloned policy alone | see the log | 4.62 m |
| 3g | plus damping and the variance term | see the log | 3.30 m |
