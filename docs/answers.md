# Answers to the written and multiple-choice questions

The original assignment asked for written answers and for multiple-choice answers. They
are here, because a portfolio repo must not hold answer keys inside the code.

The text below keeps the technical content of the original answers. The wording follows
ASD-STE100 Simplified Technical English.

The numbers in the answers of project 1 come from the original run. The port measures the
angle error with the wrap of the full circle, so it gives 0.5075 rad and 0.0126 rad. The
reasoning does not change: the sine-cosine model is much more accurate. See
[results.md](results.md).

---

## Project 1 — vision-based state estimation

**Is the model accurate? Does the training loss decrease in each step?**

The mean error across 3 runs is 0.5301 rad, or approximately 30 degrees. That is not
accurate. The training loss decreases, but not in each step. Near epoch 30 to 40 the loss
increases a little, and then it decreases again. The cause is the optimizer: SGD uses a
small batch in each step, so the gradient is noisy.

**Why is a separate test set necessary?**

The test set shows how well the model works on data that it did not see. Without it, you
cannot find overfit. A model can learn the training set and still fail on new data.

**Why do the loss values of the two models differ so much?**

The loss of `M_trig` starts near 0.50 and decreases to approximately 0.0003. The loss of
`M_theta` starts near 10 and decreases to approximately 0.35. The models have different
output ranges. The angle covers a full circle, so the largest squared error is
(2π)² ≈ 39.5. The sine and the cosine are in [−1, 1], so the largest squared error is 4
for each output.

**Why is the indirect prediction more accurate?**

The mean test errors are 0.5301 rad and 0.0706 rad. The direct model must learn a function
with a step at ±π, because the angle wraps. Two angles that are near to each other, for
example 3.13 rad and −3.13 rad, have a difference of only 0.02 rad, but the loss function
gives them a very large penalty. The sine and the cosine are continuous functions of the
angle. They have no step, so the network learns the mapping more easily.

**Why is the sine alone not sufficient?**

The sine gives the same value for two different angles, because sin(θ) = sin(π − θ). The
arcsine is unique only in [−π/2, π/2]. You must have the sine and the cosine to find the
angle without ambiguity. The `atan2` function then gives the correct quadrant.

---

## Project 2 — model-based control, LNN, ILC

All questions in project 2 are multiple-choice questions.

| Task | Question | Answer |
|---|---|---|
| 2a.1 | Why is PD on the joint angles better than PD on the link angles? | B — the dynamics of link 2 are faster, so link 2 oscillates |
| 2a.2 | What is the steady state of PD with gravity compensation? | A — θ = θ_d |
| 2a.3 | What does the term C(θ_d, θ̇_d)·θ̇_d compensate? | B — the Coriolis and centrifugal forces |
| 2a.3 | Why does a pure feedforward controller become unstable? | B and C |
| 2a.4 | Which equation describes the closed loop of PD+? | C — M(θ)(θ̈ − θ̈_d) + C(θ, θ̇)(θ̇ − θ̇_d) = kp·e + kd·ė |
| 2c.1 | What occurs if the dataset has almost no torque? | B — the potential energy and the inertia cannot be separated |
| 2c.1 | What occurs if the torque is much larger than gravity? | C — the potential energy is not learned accurately |
| 2d.1 | Which is the correct nonlinear state-space form? | B |
| 2d.1 | How are δx, δτ and the equilibrium torque defined? | D |
| 2d.1 | Which equation gives the error dynamics? | C |
| 2d.1 | What are the analytical B, C and D matrices? | A — B = [0; M⁻¹], C = [I 0], D = 0 |
| 2d.3 | What occurs if S_lq becomes larger? | C — the convergence becomes slower |
| 2d.3 | What occurs if Q_lq and S_lq are multiplied by the same factor? | A — there is no effect |

---

## Project 3 — Gaussian processes and behavioural cloning

**Task 3b.7 — constant mean against zero mean.**

The GP with a constant mean learns an offset and moves all predictions by that offset.
This helps, because the mean compressive strength of the concrete is not near zero. The
constant-mean model therefore has a smaller error.

For safety, the zero-mean model is better. Far from the data it predicts a value near
zero, so it gives too small a strength. The constant-mean model predicts a larger value
in the same area, and its lower confidence bound can stay above the true strength. Such a
prediction is unsafe, because it says that the concrete is stronger than it is.

**Task 3d.5 — what the phase portraits show.**

The colour shows the uncertainty. A bright colour is a large standard deviation, and a
dark colour is a small standard deviation.

The large-oscillation dataset covers the full configuration space from −π to π, so the
uncertainty stays small everywhere. The small-oscillation dataset covers only a small area
near −π/2. On the right side of that plot the uncertainty becomes large.

The flow lines on the right side of the second plot are horizontal. The GP has a zero
mean, so it predicts an angular acceleration of zero where it has no data. The vertical
component of the vector field is therefore zero, and only the horizontal component, the
angular speed, remains.

**Task 3e.4 — the two MLP phase portraits.**

The two datasets cover different areas of the state space. With the large-oscillation
data the MLP learns the dynamics everywhere, and the vector field agrees with the physics.
With the small-oscillation data the MLP never saw the right side of the state space. Unlike
the GP, which returns to its prior, the MLP extrapolates. Its predictions in that area can
disagree with the physics.

**Task 3e.5 — the MLP against the GP.**

With full data the two models agree. Without data they do not. The GP predicts zero and
shows a large uncertainty. The MLP predicts an arbitrary value and shows nothing. The GP
gives an uncertainty at each point, so it is more useful for a safety-critical controller.

**Task 3f.4 — the effect of the periodic kernel.**

Yes, you can see the effect in the surface plot. The predicted torques repeat with a
period of 2π in both angles. This is correct, because a joint angle is periodic. The GP
therefore generalises to angles that are outside the training data.

**Task 3f.5 — why the cloned controller fails.**

The GP receives only the two angles. It receives no speed and no reference. The original
PD controller uses the position error and the speed error. The GP therefore learned the
torque as a function of the angle alone, and that function is correct only near the
training path. When the robot moves away from the path, the GP applies a torque that
belongs to a different point of the path. The error becomes larger, and the system
diverges.

**Task 3f.6 — why the variance term makes the system stable.**

The uncertainty of the GP is small along the training path and large away from it. The
extra torque is −k_var · σ · sign(∇σ). It pushes the robot in the direction where the
uncertainty decreases, and that direction points to the training path. This gives a
feedback to the path, although the controller does not know the desired position. The
uncertainty of the GP is a map of the place where the data was collected. To minimise it
is to stay on the path.

**Task 3g.5 — why the delta-angle controller fails.**

The GP predicts the change of the angle as a function of the current angle. A gain then
makes a torque from that change. The controller does not track the path. When the robot
moves away from the path, the GP gives the change that belongs to the nearest training
point, and that change can push the robot further away. There is no speed term and no
error term, so the robot is sensitive to a disturbance and cannot return.

**Task 3g.7 — why the performance becomes better.**

There are two causes. First, the negative variance gradient moves the robot to the area
with a small variance, and that area is the reference path. Second, the damping term Kd
decreases the oscillation. The pure GP controller has no speed feedback, so it oscillates.

**Task 3h.1 — behavioural cloning with the angular speed as an input.**

If the initial state has an error, the system diverges more quickly, because the GP has no
speed information. You can add the angular speed as an input. The GP then learns the map
from the angles and the speeds to the torque, and it acts like a PD controller. This makes
the controller more robust. But the input space becomes larger, so you need more data.

**Task 3h.2 — thoughts about the minimisation of the variance.**

The gradient of the GP variance points in the direction where the uncertainty increases.
The negative gradient therefore points to the data. This moves the robot to the training
path.

This works for the robot arm, because the data was collected along one closed path. The
variance has one clear valley along that path, and it is large away from it.

This does not work for every dynamic system. A vehicle collects data on many different
paths. The variance then has many local minima, and the vehicle can move to the wrong
one. Also, the noise on the gradient increases with the number of states, so the direction
becomes unreliable in a large state space.

**Task 3h.3 — a comparison of the safety properties.**

Condition 1: a person is against the robot and cannot move away.

- PD with gravity compensation is the least safe. It reads the resistance as a position
  error and increases the torque. The force on the person increases.
- Torque cloning in the configuration space is safer. It does not increase the torque with
  the error. But it also does not know about the contact, so it continues to push.
- Reference-trajectory cloning is the safest. The GP gives a small change of the angle,
  and the torque is proportional to that change. The force stays small.

Condition 2: the person pushes the robot away.

- PD with gravity compensation resists strongly. The force increases with the deviation.
  For a large deviation this is dangerous.
- Torque cloning moves to a state with a larger uncertainty. The variance term then moves
  it back. The reaction is softer than PD, but the path depends on the variance map.
- Reference-trajectory cloning stays the safest. It gives a small change of the angle for
  the new state, and the robot returns to the path when the person releases it.
