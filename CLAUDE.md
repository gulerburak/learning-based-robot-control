# Instructions for Claude

## Rules for this repo

1. Write all text in **ASD-STE100 Simplified Technical English**. This applies to chat
   replies, code comments, docstrings, commit messages and documents. Use short sentences,
   the active voice, and simple words.
2. **Do not over-comment the code.** Write a comment only when the reason is not clear
   from the code. Keep short docstrings that give the units and the array shapes.
3. **Never add a `Co-Authored-By` line or any other agent trailer to a commit.**

## What this repo is

Three projects that show robot state estimation, robot control, and learning for control.
The work comes from the practical assignment of the TU Delft course RO47019 "Intelligent
Control Systems" (2025-2026 Q3). The original work was in Jupyter notebooks that nbgrader
controls. This repo holds the same work as plain Python.

**The original notebooks are at `/home/nuke/code/ics-pa-sv`.** Read `docs/source-map.md`
before you look at them, because the notebooks are large and hold stored figures.

## Repo map

| Path | Content |
|---|---|
| `jax_double_pendulum/` | The simulator of the course. It is copied, not changed. MIT licence, TU Delft. |
| `p1_vision_state_estimation/` | CNN that finds a pendulum angle in an image. PyTorch. |
| `p2_control/` | PD control, Lagrangian Neural Network, iterative learning control. JAX, Flax, Optax. |
| `p3_gp_learning/` | Gaussian processes, and behavioural cloning of a controller. GPyTorch. |
| `docs/` | The research notes. Read these first. |
| `data/` | The small datasets. The large ones are made at run time. |
| `outputs/` | Figures and animations. Not in git. |

`p3_gp_learning` imports `p2_control.controllers`. `p2_control` and `p3_gp_learning`
import `jax_double_pendulum`.

## Documents

| File | Content |
|---|---|
| `docs/source-map.md` | Which notebook cell became which file. Also the faults that the port corrects. |
| `docs/results.md` | Every number from the original runs. These are the targets. |
| `docs/answers.md` | The written answers and the multiple-choice answers. |
| `docs/01-…`, `02-…`, `03-…` | One page for each project: the problem, the method, the result. |

## How to run

```bash
pip install -e .

python -m p2_control.tasks.pd_control            # seconds
python -m p3_gp_learning.tasks.gp_tensile        # seconds
python -m p1_vision_state_estimation.run --epochs 2 --runs 1
```

Every task module has an `argparse` interface. Use `--help`. The long jobs
(`collect_dataset`, `train_lnn`, `q_ilc`) have flags that make them shorter.

### Long jobs in the background

`train_lnn` needs approximately one hour, and the full project 1 run needs approximately
40 minutes. Start such a job with `setsid`, or the agent harness stops it when it cleans
up the process group of the tool call:

```bash
setsid nohup bash -c "python3 -m p2_control.tasks.train_lnn > train.log 2>&1" \
    < /dev/null > /dev/null 2>&1 &
```

The progress bar writes `\r`, so read the log with `tr '\r' '\n' < train.log | tail`.

## Rules for the code

- Set the JAX flags in `p2_control/__init__.py` before you make an array. The ILC tasks
  need `jax_enable_x64`.
- Do not use `ipynb.fs.full`, `tqdm.notebook`, or `%matplotlib`. They belong to notebooks.
- Keep the tuned values in module constants, not in the middle of a function.

## Port status

The port is complete. `docs/results.md` holds a table that compares a run of this repo
against the values of the original notebooks. All control results and all GP
hyperparameters agree. Task 3b differs, because the inducing points start at different
places; the note in that file gives the detail.
