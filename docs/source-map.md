# Source map

This page maps the original notebooks to the files in this repo. Use it when you must
find the origin of a function, or when you compare the port against the source.

Source repo: `/home/nuke/code/ics-pa-sv` (TU Delft RO47019, nbgrader template).
The notebooks hold the course scaffolding, the hidden tests, and the answers. The answers
are the lines below each `# YOUR CODE HERE` marker. The notebooks have no
`### BEGIN SOLUTION` markers, except in problem 1.

## How to read a notebook without the image outputs

The notebooks are large, because they hold the stored figures. Read the source cells only:

```bash
python3 -c "
import json, sys
nb = json.load(open(sys.argv[1]))
for i, c in enumerate(nb['cells']):
    print(f'#--- cell {i} {c[\"cell_type\"]} ---')
    print(''.join(c['source']))
" assignment/problem_2/lnn.ipynb
```

---

## Project 1 — vision-based state estimation

Source: `assignment/problem_1/`. The notebooks import no repo code. They use PyTorch only.

| Source | Cell | Content | New file |
|---|---|---|---|
| `task_1a_extract_dataset.ipynb` | 5–15 | Unpacks the zip, makes a GIF | `p1_vision_state_estimation/make_dataset.py` |
| `task_1b_train_neural_network.ipynb` | 8 | `CNNDataset` | `dataset.py` → `PendulumImageDataset` |
| " | 10 | Transform chain | `dataset.py` → `build_transform()` |
| " | 12 | `load_dataloaders` | `dataset.py` → `load_dataloaders()` |
| " | 16 | `evaluate_model` | `train.py` → `evaluate_model()` |
| " | **19** | `CNNTheta` | `models.py` → `CNNTheta` |
| " | **22** | Training loop, theta | `train.py` → `train_model()` |
| " | **31** | `CNNTrig` | `models.py` → `CNNTrig` |
| " | **34** | Training loop, trig | `train.py` → `train_model()` |
| " | 24, 36 | Evaluation across runs | `train.py` → `evaluate_across_runs()` |
| " | 26, 28, 39, 41, 43 | Written answers | `docs/answers.md` |

Bold cells hold the student code.

### Changes in the port

1. **Fault fix.** Cell 34 calls `model_theta.eval()` in the trig loop. It must be
   `model_trig.eval()`. The fault had no effect, because the networks have no dropout
   layer and no batch-norm layer, and `evaluate_model` runs under `torch.no_grad()`.
2. **Fault fix.** The notebook calls `load_dataloaders` one time, before
   `manual_seed(run)`. The split therefore depends on the global random state. The port
   sets the seed first, then makes the split. The numbers move a little, but they are
   repeatable.
3. **Addition.** The task asked for a check against the validation set in each epoch. The
   notebook makes `val_loader` but never uses it. The port adds the check.
4. **Addition.** The port makes a loss-curve figure and an error-against-angle figure. The
   notebook printed the loss only.

---

## Project 2 — model-based control, LNN, ILC

Source: `assignment/problem_2/`. Four "notebooks as modules" hold the library code. The
task notebooks import them with `from ipynb.fs.full.<name> import ...`. In this repo they
are plain modules.

| Source notebook | Function or class | New file |
|---|---|---|
| `controllers.ipynb` | `ctrl_fb_pd`, `ctrl_fb_pd_rel`, `ctrl_ff_gravity_compensation`, `ctrl_ff_feedforward`, `ctrl_ff_pd_plus` | `p2_control/controllers.py` |
| `linearization.ipynb` | `continuous_state_space_dynamics`, `continuous_linear_state_space_representation_autograd`, `cont2discrete_zoh`, `linearized_discrete_forward_dynamics`, `closed_loop_fb_continuous_forward_dynamics`, `linearize_closed_loop_fb_system_about_trajectory` | `p2_control/linearization.py` |
| `lnn.ipynb` | `MassMatrixNN`, `PotentialEnergyNN`, `kinetic_energy_fn`, `potential_energy_fn`, `lagrangian_fn`, `mass_matrix_fn`, `dynamical_matrices`, `continuous_forward_dynamics`, `continuous_state_space_dynamics`, `discrete_forward_dynamics` | `p2_control/lnn.py` |
| `lnn_training.ipynb` | `load_datasets`, `create_learning_rate_fn`, `initialize_train_states`, `mse_loss_fn`, `compute_metrics`, `make_vectorized_discrete_forward_dynamics`, `train_step`, `eval_step`, `train_epoch`, `eval_model`, `run_lnn_training` | `p2_control/lnn_training.py` |
| `ilc.py` (given) | `init_ilc_its`, `apply_ilc_control_action_to_system` | `p2_control/ilc.py` (copied) |
| `ilc_analysis.py` (given) | ILC figures and animations | `p2_control/ilc_analysis.py` (copied) |
| `lnn_analysis.py` (given) | Dataset and LNN figures | `p2_control/lnn_analysis.py` (copied) |

| Task notebook | Content | New file |
|---|---|---|
| `task_2a-1_pd_control.ipynb` | PD in link space and joint space, `kp`, `kd` | `tasks/pd_control.py` |
| `task_2a-2_…gravity_compensation…` | PD with gravity compensation | `tasks/pd_gravity_compensation.py` |
| `task_2a-3_…feedforward…` | PD with inverse-dynamics feedforward | `tasks/pd_feedforward.py` |
| `task_2a-4_pd_plus…` | PD+ (Paden–Panja) | `tasks/pd_plus.py` |
| `task_2c-1_collect_dataset.ipynb` | `sample_system_evolution`, `save_sim_data_to_dataset` | `tasks/collect_dataset.py` |
| `task_2c-2_implement_lnn.ipynb` | Tests only, no student code | `tests/test_lnn.py` |
| `task_2c-3_train_lnn.ipynb` | Training run, best epoch | `tasks/train_lnn.py` |
| `task_2c-4_rollout_learned_dynamics…` | Rollout against the true model | `tasks/rollout_lnn.py` |
| `task_2c-5_control_with_learned_dynamics…` | PD+ with the learned matrices | `tasks/control_with_lnn.py` |
| `task_2d-1_linearization.ipynb` | Open-loop linearization and rollout | `tasks/linearize.py` |
| `task_2d-2_pd_ilc.ipynb` | `blk_diag`, `compute_pd_ilc_gains`, `learning_rule_pd_ilc`, `run_pd_ilc`, `pd_ilc_iteration` | `p2_control/ilc_pd.py` + `tasks/pd_ilc.py` |
| `task_2d-3_q_ilc.ipynb` | `compute_lifted_system_input_to_output_mapping`, `compute_lqr_optimal_gains`, `learning_rule_q_ilc`, `run_q_ilc`, `q_ilc_iteration` | `p2_control/ilc_q.py` + `tasks/q_ilc.py` |

### Changes in the port

1. `kinetic_energy_fn` had a debug line `print("T", T.shape)`. It is removed.
2. `linearize_closed_loop_fb_system_about_trajectory` stays un-jitted. Its `@jit` was
   commented out in the source, because `jax.scipy.linalg.expm` runs under `vmap`.
3. `tqdm.notebook` becomes `tqdm`. `%matplotlib widget` becomes a file output.
4. The multiple-choice answers move from `answer_N` variables into `docs/answers.md`.
5. The tuned gains move into module constants, so that one file holds them.

---

## Project 3 — Gaussian processes and behavioural cloning

Source: `assignment/problem_3/`. It imports the controllers of problem 2.

| Source notebook | Content | New file |
|---|---|---|
| `utils.py` (given) | `process_data`, `plot_data`, `generate_training_data`, `split_2d_columns` | `p3_gp_learning/data_utils.py` |
| `task_3a_…gp_single_input` | `ExactGPModel`, `SVGPModel`, both training loops | `gp_models.py`, `tasks/gp_tensile.py` |
| `task_3b_…multiple_inputs` | ARD `SVGPModel`, `evaluate_model`, mean comparison | `gp_models.py`, `tasks/gp_concrete.py` |
| `task_3c_dataset_generation…` | Three robot datasets (no student code) | `tasks/make_robot_datasets.py` |
| `task_3d_fit_forward_dynamic_with_gp` | `MultitaskGPModel` (Matérn), `ForwardDynamicsModel`, phase portraits | `gp_models.py`, `wrappers.py`, `tasks/dynamics_gp.py` |
| `task_3e_fit_forward_dynamic_with_mlp` | MLP and its trainer | `mlp_model.py`, `tasks/dynamics_mlp.py` |
| `task_3f_behavioural_cloning_torques…` | Periodic `MultitaskGPModel`, `ControllerModel`, `ctrl_fb` with variance repulsion | `gp_models.py`, `wrappers.py`, `cloning.py`, `tasks/clone_torques.py` |
| `task_3g_behavioural_cloning_reference…` | Delta-angle cloning, damping | `wrappers.py`, `cloning.py`, `tasks/clone_reference.py` |
| `task_3h_open_questions` | Three written answers | `docs/answers.md` |

### Changes in the port

1. `from ipynb.fs.full.controllers import …` becomes `from p2_control.controllers import …`.
2. Task 3f ended with a `ValueError` in `animate_robot`, because the simulation array and
   the reference array had different lengths (55 against 60, and 875 against 900). The
   port cuts both to the same length.
3. `ForwardDynamicsModel.train` and `ControllerModel.train` put a tensor that needs a
   gradient into a numpy array. The port calls `.item()`. This removes the warning.
4. The `Slider` widget in task 3b becomes a static figure.
5. The written answers move into `docs/answers.md`.
