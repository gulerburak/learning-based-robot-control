# Learning-based robot control

Three projects on one 2-link robot arm. They answer three questions:

1. **Where is the robot?** A convolutional network reads the joint angle from a camera
   image.
2. **How does the robot move?** A neural network learns the physics of the robot, and a
   learning controller corrects a model that is wrong.
3. **How sure is the model?** A Gaussian process reports its own uncertainty, and a
   controller uses that uncertainty as a feedback signal.

Every result below comes from a script in this repo. All text follows ASD-STE100
Simplified Technical English.

---

## Results

| Project | What it shows | Result |
|---|---|---|
| **1. Vision** | The output representation of an angle changes the accuracy more than the network size. | **0.071 rad** with sine and cosine, against **0.530 rad** with the angle. The models differ by 31 parameters. |
| **2. Control** | A Lagrangian Neural Network learns the dynamics well enough to replace the physical model in the controller. | **0.049 m** tip error with the learned model, against 0.049 m with the true model. |
| **2. ILC** | Iterative Learning Control removes the effect of a model error of 80 %. | **0.034 m** tip error, which is better than a PD controller with the correct model. |
| **3. GP** | The variance of a Gaussian process makes an unstable cloned controller stable. | **0.40 m** largest path error, against 2.08 m without the variance term. |

---

## 1. Vision-based state estimation

`p1_vision_state_estimation/` · PyTorch · [full page](docs/01-vision-state-estimation.md)

A CNN with 8000 parameters reads the angle of a pendulum from a 24x24 image. Two output
representations are compared:

* **Direct** — the network gives the angle. Error: 0.5301 ± 0.2476 rad.
* **Indirect** — the network gives the sine and the cosine, and `atan2` gives the angle.
  Error: 0.0706 ± 0.0322 rad.

The direct model must learn a function with a step at ±π, because the angle wraps there.
Two angles that are 0.02 rad apart get a penalty of 6.26 rad. The sine and the cosine have
no step, so the same network becomes 7.5 times more accurate.

## 2. Model-based control, learned dynamics, iterative learning control

`p2_control/` · JAX, Flax, Optax · [full page](docs/02-model-based-and-learned-control.md)

The robot must follow an ellipse. It starts away from the path.

**Classical control.** PD in the link space and in the joint space, PD with gravity
compensation, PD with inverse-dynamics feedforward, and PD+ (Paden–Panja). PD+ reaches
0.0491 m with gains that are ten times smaller than the gains of the simple PD controller,
because its feedforward term makes the error dynamics linear.

**A Lagrangian Neural Network.** Instead of the acceleration, the network learns the
energy. One network gives the mass matrix through a Cholesky form, so the matrix is always
positive definite. A second network gives the potential energy. Automatic differentiation
of the Lagrangian gives M, C and G, and an RK4 step gives the next state. The loss goes
through all of it.

After training on 249750 samples the learned model replaces the physical model in the
PD+ controller. The tip error is 0.0493 m, and the true model gives 0.0491 m.

**Iterative Learning Control.** The controller now uses a **wrong** model: its masses are
too large by a factor of 1.8. The closed loop is linearized along the path with
forward-mode automatic differentiation, discretized with a zero-order hold, and stacked
into a lifted system of 1998 × 1998. Q-ILC then computes the learning gains from an LQR
problem. After 1000 runs the tip error is 0.0335 m, although the model stays wrong.

## 3. Gaussian processes and behavioural cloning

`p3_gp_learning/` · GPyTorch · [full page](docs/03-gaussian-processes-and-cloning.md)

**What a GP gives.** An exact GP and a sparse GP fit a tensile test of steel. An ARD model
predicts the strength of concrete and finds by itself that the age of the sample has the
largest effect. The comparison of a constant mean against a zero mean is about safety: far
from the data, a zero mean warns and a constant mean does not.

**Where a GP is better than an MLP.** Both learn the dynamics of the robot. Where there is
no training data, the GP falls back to its prior and reports a large standard deviation.
The MLP gives a confident answer that disagrees with the physics.

**Uncertainty as feedback.** A GP with a periodic kernel clones a PD controller. It sees
the angles only, so the closed loop diverges. The variance of the GP is small on the
training path and large away from it. A torque along the negative gradient of the variance

    tau_repel = -k_var * sigma * sign(d(sigma^2) / d(theta))

moves the robot back to the path. The largest path error falls from 2.08 m to 0.40 m. The
controller gets a feedback to the path, although it never knows the desired position.

A second experiment clones the path itself, without a teacher controller. There the same
idea helps (4.62 m to 3.30 m), but the controller stays weak. The
[full page](docs/03-gaussian-processes-and-cloning.md) explains why.

---

## Install and run

```bash
git clone git@github.com:gulerburak/learning-based-robot-control.git
cd learning-based-robot-control
pip install -e .
```

Python 3.10 to 3.13. A GPU is not necessary. The versions of JAX and Flax are pinned,
because the ILC tasks need 64-bit floats.

```bash
# Project 2 — seconds each
python -m p2_control.tasks.pd_control
python -m p2_control.tasks.pd_gravity_compensation
python -m p2_control.tasks.pd_plus
python -m p2_control.tasks.linearize

# Project 3 — minutes
python -m p3_gp_learning.tasks.gp_tensile
python -m p3_gp_learning.tasks.make_robot_datasets
python -m p3_gp_learning.tasks.dynamics_gp
python -m p3_gp_learning.tasks.clone_torques

# Project 1 — make the images first
python -m p1_vision_state_estimation.make_dataset --render
python -m p1_vision_state_estimation.run

# Long jobs. Each has flags that make it shorter.
python -m p2_control.tasks.collect_dataset   # approximately 5 minutes
python -m p2_control.tasks.train_lnn         # approximately 1 hour on a CPU
python -m p2_control.tasks.pd_ilc            # approximately 3 minutes
python -m p2_control.tasks.q_ilc             # approximately 10 minutes
```

All scripts have an `argparse` interface. Use `--help`. The figures go into `outputs/`.

```bash
pip install -e ".[dev]"
python -m pytest tests/test_lnn.py
```

The test checks the Lagrangian Neural Network against reference values of the course.

---

## Repo map

| Path | Content |
|---|---|
| `p1_vision_state_estimation/` | The dataset, the two CNNs, the training loop. |
| `p2_control/` | The controllers, the Lagrangian Neural Network, the linearization, PD-ILC and Q-ILC. `tasks/` holds the runnable scripts. |
| `p3_gp_learning/` | The GP models, the MLP, the cloned controllers. `tasks/` holds the runnable scripts. |
| `jax_double_pendulum/` | The robot simulator. See the credit below. |
| `docs/` | One page for each project, the recorded results, and the written answers. |
| `data/` | The small datasets. See [data/README.md](data/README.md). |
| `tests/` | The reference test of the Lagrangian Neural Network. |

---

## Credit

The folder `jax_double_pendulum/` is the simulator of the course RO47019 "Intelligent
Control Systems" at Delft University of Technology. It gives the dynamics, the kinematics,
the trajectory generator and the plot helpers. It is copied without change, and it is MIT
licensed. Authors: Maximilian Stölzle, Chuhan Zhang, Lorenzo Lyons, Giovanni Franzese,
Tomás Coleman and Jingyue Liu. Source:
[tud-phi/ics-pa-sv](https://github.com/tud-phi/ics-pa-sv).

Some plot helpers in `p2_control/` (`ilc.py`, `ilc_analysis.py`, `lnn_analysis.py`) also
come from the course. `docs/source-map.md` records what came from where.

## A note about the course

This work started as the practical assignment of RO47019 at TU Delft. The course still
runs. **If you follow that course, do not copy this code.** You will learn nothing, and
your school has rules against it. Read `docs/` instead, and then write your own solution.

## Licence

MIT. See [LICENSE](LICENSE).
