# Datasets

## In this folder

| File | Size | Content | Source |
|---|---|---|---|
| `concrete_data.csv` | 57 kB | 1030 concrete mixtures. 8 inputs (cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age) and the compressive strength in MPa. | [UCI Machine Learning Repository, Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) |
| `tensile_strength.txt` | 20 kB | 373 measurements of a tensile test on steel. Position, load, strain and time. | [University of Illinois, Mechanical Testing Instructional Lab](https://mtil.illinois.edu/DATA/_DataAnalysisHELP/_Tensile_Example/Tensile_Analysis/6150_Tensile_Data/) |

The file `tensile_strength.txt` has an unusual format. It uses `\r` as the line end, and
the first line is junk. Read it with `pd.read_csv(path, sep="\t", header=1)`. The last
row is the break of the sample, so the code removes it.

## Made at run time

These datasets are not in git. Make them with the scripts.

| Folder | Made by | Content |
|---|---|---|
| `pendulum_images/` | `python -m p1_vision_state_estimation.make_dataset --render` | 3600 images of a pendulum, 500x500x3, with the angle as the label. |
| `p2/` | `python -m p2_control.tasks.collect_dataset` | 249750 transitions of the robot dynamics, for the Lagrangian Neural Network. |
| `p3/` | `python -m p3_gp_learning.tasks.make_robot_datasets` | Three tables of robot runs: a small oscillation, a large oscillation, and a run with a PD controller. |

### The pendulum images

The images come from the `Pendulum-v1` environment of gymnasium. The angle goes over the
full circle in steps of 0.1 degrees. The angle is zero when the link points up, and it
increases counterclockwise.

```bash
pip install -e ".[render]"
python -m p1_vision_state_estimation.make_dataset --render --gif
```

If you already have an archive of the images, use it instead. It gives the same files:

```bash
python -m p1_vision_state_estimation.make_dataset --zip path/to/pendulum_dataset.zip
```
