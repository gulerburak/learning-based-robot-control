# Project 2 — Model-based control, learned dynamics, iterative learning control

**Code:** `p2_control/` · **Libraries:** JAX, Flax, Optax

The robot is a planar 2-link pendulum. The task is always the same: the tip must follow an
ellipse. The robot starts away from the path, so the controller must correct an error.

The project answers one question in three steps: **what do you do when you do not have a
good model of the robot?**

---

## Step 1 — Control with a good model

| Controller | Feedforward term | RMSE of the tip |
|---|---|---|
| PD on the link angles | none | 0.0635 m |
| PD on the joint angles | none | 0.0501 m |
| PD + gravity compensation | G(θ) | **0.0269 m** |
| PD + inverse dynamics | M(θᵈ)θ̈ᵈ + C(θᵈ,θ̇ᵈ)θ̇ᵈ + G(θᵈ) | 0.0619 m |
| PD+ (Paden–Panja) | M(θ)θ̈ᵈ + C(θ,θ̇)θ̇ᵈ + G(θ) | 0.0491 m |

Two results are worth attention.

The joint-space PD controller is better than the link-space PD controller, although the
gains are the same. The torque acts on the joints, so the error must be measured there.

The PD+ controller reaches 0.0491 m with gains that are **ten times smaller** than the
gains of the simple PD controller (500 against 5000). The feedforward term makes the
error dynamics linear, so the feedback term has little work to do. A small gain means a
soft robot, and a soft robot is safer near a person.

```bash
python -m p2_control.tasks.pd_control
python -m p2_control.tasks.pd_gravity_compensation
python -m p2_control.tasks.pd_plus
```

---

## Step 2 — Learn the model from data (Lagrangian Neural Network)

A network can learn the acceleration directly from (θ, θ̇, τ). Such a network fits the
data, but it does not obey the physics. Its energy grows or falls without a cause, and a
rollout over 10 s diverges.

The Lagrangian Neural Network learns the **energy** instead:

| Part | Network | Note |
|---|---|---|
| Mass matrix M(θ) | 4 layers, 32 units, softplus | The network gives the lower triangle L, and M = L·Lᵀ. M is therefore always symmetric and positive definite. |
| Potential energy U(θ) | 4 layers, 32 units, softplus | One scalar. |

The kinetic energy is T = ½ θ̇ᵀ M(θ) θ̇, and the Lagrangian is L = T − U. Automatic
differentiation then gives the equation of motion:

* M is the Hessian of L against θ̇.
* C comes from the Christoffel symbols, which need ∂M/∂θ.
* G is the gradient of U.

The loss compares the angular speed at the next time step. The prediction comes from one
RK4 step, so the gradient goes through the integrator and through all of those
derivatives.

**Training:** 249750 samples from 250 random rollouts, 250 epochs, AdamW, cosine decay
with a warmup.

**Result:** the validation loss falls to 2.44e-07. Then:

| Test | RMSE of the tip |
|---|---|
| Free rollout over 10 s against the true model | **0.0392 m** |
| PD+ control that uses the learned M, C and G | **0.0493 m** |

The controller with the learned model is as good as the controller with the true model
(0.0491 m). The network learned the physics, not only a fit of the data.

```bash
python -m p2_control.tasks.collect_dataset   # approximately 5 minutes
python -m p2_control.tasks.train_lnn         # approximately 1 hour on a CPU
python -m p2_control.tasks.rollout_lnn
python -m p2_control.tasks.control_with_lnn
```

---

## Step 3 — Learn the correction, not the model (Iterative Learning Control)

Sometimes you cannot learn a model, but the robot repeats the same path many times. ILC
then learns a torque correction from the error of the last run.

To show that this works, the controller uses a **wrong** model on purpose. Its masses and
inertias are too large by a factor of 1.8 to 3.

The method needs a linear model along the path:

1. Write the closed loop (robot **and** its PD controller) in the state-space form.
2. Take the Jacobian at every time step with forward-mode automatic differentiation.
3. Discretize with a zero-order hold. This uses the matrix exponential of the block
   matrix [[A, B], [0, 0]].
4. Stack all time steps into one large "lifted" system. The matrix P maps the full torque
   series to the full output series. It is block lower triangular, because a torque
   cannot change the past. With 1000 time steps it has a size of 1998 × 1998.

| Method | Learning gains | Iterations | RMSE of the tip |
|---|---|---|---|
| PD-ILC | two scalars, tuned by hand | 500 | **0.0586 m** |
| Q-ILC | from an LQR problem | 1000 | **0.0335 m** |

Q-ILC gives the gain matrix from a cost function:

    min over U of   Eᵀ Q E + ΔUᵀ S ΔU
    L_opt = (Pᵀ Q P + S)⁻¹ Pᵀ Q

`Q` gives the weight of the tracking error and `S` gives the weight of the torque change.
A large `S` makes the convergence slower and the torque smoother. If both are multiplied
by the same factor, nothing changes.

The result is the important part: the model of the controller stays wrong, but the tip
error falls to 0.0335 m. That is better than the PD controller with the **correct** model.

```bash
python -m p2_control.tasks.linearize
python -m p2_control.tasks.pd_ilc     # approximately 3 minutes
python -m p2_control.tasks.q_ilc      # approximately 10 minutes, P goes into a cache
```

---

## Test

`tests/test_lnn.py` checks the network against reference values of the course. It tests
the construction of the mass matrix and all derivatives of the Lagrangian.

```bash
python -m pytest tests/test_lnn.py
```
