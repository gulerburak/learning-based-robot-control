# Project 3 — Gaussian processes and behavioural cloning

**Code:** `p3_gp_learning/` · **Library:** GPyTorch

A neural network gives an answer everywhere, and it gives no warning when it does not
know. A Gaussian process gives an answer **and** a variance. This project shows what that
variance is good for, and ends with a controller that uses its own uncertainty as a
feedback signal.

---

## Part 1 — What a Gaussian process gives

### A tensile test of steel (one input)

The data is a tensile test: the load against the position of the clamp. The curve has an
elastic part, a yield point, and then plastic flow.

| Model | Lengthscale | Noise |
|---|---|---|
| Exact GP, all 373 samples | 0.888 | 0.0009 |
| Sparse GP, 10 inducing points | 2.36 | 0.0303 |

The sparse model cannot follow the sharp yield point with 10 points. It therefore makes
the function smoother and calls the difference "noise". This is the cost of the
approximation, and the model reports it.

```bash
python -m p3_gp_learning.tasks.gp_tensile
```

### The strength of concrete (eight inputs)

1030 concrete mixtures. The model predicts the compressive strength from the mixture and
the age. The kernel has ARD, so each input gets its own lengthscale. A small lengthscale
means a large effect.

The age of the sample gets the smallest lengthscale by a large distance. The model found
by itself that concrete becomes stronger with time.

The interesting measure is not the error. It is the **unsafe share**: how often the lower
2σ bound of the model is still above the true strength. Such a prediction says that the
concrete is stronger than it is, and a person could build with it.

| Mean function | MAE on the test set | Calibration | Unsafe share |
|---|---|---|---|
| Constant | 5.26 MPa | 0.80 | 0.034 |
| Zero | 5.84 MPa | 0.77 | 0.044 |

A constant mean fits better, because the strength of concrete is not near zero. Far from
the data, however, a zero mean falls back to zero and warns, and a constant mean keeps a
large value.

> **Note on the numbers.** The original notebook measured 6.35 MPa and 6.84 MPa, and it
> found the zero mean safer (0.058 against 0.087). The port chooses its 300 inducing
> points with a different random function, so the two runs are not identical. The
> ordering of the error stays the same in both. The unsafe share is a small count on 206
> test samples, so it moves between runs. Do not read one run as proof.

```bash
python -m p3_gp_learning.tasks.gp_concrete
```

---

## Part 2 — Where a GP is better than a neural network

Both models learn the same map: the state and the torque of the robot give the angular
accelerations. Two datasets are used:

* **large oscillation** — the robot swings over the full circle, so the data covers the
  whole state space.
* **small oscillation** — the robot hangs down and swings a little, so the data covers a
  small area near −π/2 only.

The result is a phase portrait. The flow lines give the motion. For the GP the colour
gives the standard deviation.

With the large-oscillation data both models agree, and both agree with the physics.

With the small-oscillation data they do not:

| | Outside the training data |
|---|---|
| **GP** | The mean falls back to the zero prior. The predicted acceleration is zero, so the flow lines become horizontal. The colour becomes bright, and that is a clear warning. |
| **MLP** | It extrapolates with its last linear pieces. The flow lines look reasonable, but they disagree with the physics. There is no warning. |

For a controller this is the whole difference. The GP tells you where you can trust it.

Learned ARD lengthscales of the GP on the large-oscillation data:
`[7.73, 7.10, 0.80, 0.86, 0.69, 0.69]` for
(θ̇₁, θ̇₂, θ₁, θ₂, τ₁, τ₂). The speeds have long lengthscales, so the acceleration changes
slowly with the speed. The angles have short lengthscales, because gravity changes quickly
with the angle.

```bash
python -m p3_gp_learning.tasks.make_robot_datasets   # run first
python -m p3_gp_learning.tasks.dynamics_gp
python -m p3_gp_learning.tasks.dynamics_mlp
```

---

## Part 3 — Uncertainty as a feedback signal

This is the main result of the project.

### The problem

A GP clones a PD controller. It learns (θ₁, θ₂) → (τ₁, τ₂) from one run of the PD
controller along the ellipse. The GP then replaces the PD controller.

The closed loop diverges. The reason is clear: the GP sees the angles only. It sees no
speed and no reference. It therefore learned a torque that is correct **on** the training
path and nowhere else. When the robot moves away, the GP applies a torque that belongs to
a different point of the path, and the error grows.

The kernel is periodic with a period held at 2π, because a joint angle is periodic. The
model learns a period of 6.2760 rad, and 2π is 6.2832. The surface plot of the policy
repeats correctly outside the training range.

### The solution

The variance of the GP is small on the training path and large away from it. It is
therefore a map that points to the path. Add a torque along the negative gradient of the
variance:

    τ_repel = −k_var · σ · sign(∂σ² / ∂θ)

The gradient comes from automatic differentiation through the GP.

| Controller | Largest error of the tip |
|---|---|
| The cloned policy alone | 2.08 m — it leaves the path |
| The same policy plus the variance term (k_var = 2.0) | **0.40 m** — it stays near the path |

The controller now has a feedback to the path, although it never knows the desired
position. The uncertainty of the model does the work.

### The same idea with the path only

Task 3g removes the teacher. The GP sees the reference path only, and it learns

    (angles now) → (change of the angles over one time step)

A gain makes a torque from that change. Alone this fails, because there is still no speed
term and no gravity compensation.

| Controller | Largest error of the tip |
|---|---|
| The cloned policy alone | 4.62 m |
| Plus gravity compensation, damping and the variance term | 3.30 m |

The damping term and the variance term make the error smaller, but this controller stays
much weaker than the controller of task 3f. The cause is the signal itself: the change of
the angle over one time step is approximately 0.01 rad, so the torque is small against the
gravity torque of the arm. The gains (kp = 500, kd = 2.0, k_var = 1.0) are the values that
the original work tuned.

This is the honest limit of the method. A controller that clones a path without a teacher
and without a speed measurement gets a weak signal, and no gain makes that signal larger
without noise.

```bash
python -m p3_gp_learning.tasks.clone_torques
python -m p3_gp_learning.tasks.clone_reference
```

---

## Where this does not work

The variance term works here because the training data lies along one closed path. The
variance then has one clear valley, and its gradient always points to that valley.

A vehicle collects data on many paths. The variance then has many local minima, and the
negative gradient can point to the wrong one. The noise on the gradient also grows with
the number of states. The method is therefore not general. It needs data with one clear
low-uncertainty area.

The full discussion of safety is in [answers.md](answers.md).
