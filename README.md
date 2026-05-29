# Hybrid Modeling for Battery Electric Vehicle SOC Estimation

MATLAB code and curated numerical outputs for the paper:

> Tsirkinidis K, Savva C, Tournaviti M, Michailidou AV, Vlachokostas C. Hybrid modeling for state of charge estimation in Battery Electric Vehicles: Balancing Accuracy, Efficiency, and Interpretability. *International Journal of Electrical Power and Energy Systems* 2026;111894. https://doi.org/10.1016/j.ijepes.2026.111894

The workflow combines an Extended Kalman Filter (EKF), sparse Gaussian Process Regression (GPR) residual correction, and adaptive Bayesian fusion for battery state-of-charge (SOC) estimation.

## Paper

- DOI: https://doi.org/10.1016/j.ijepes.2026.111894
- Journal: *International Journal of Electrical Power and Energy Systems*
- Status: in press / article number `111894`

## Repository Structure

```text
code/matlab/
  run_reproduction_workflow.m        Ordered workflow entry point
  preprocessing/                     Raw-data preprocessing and compact configs
  feature_selection/                 Deployable feature screening
  model_training/                    Final hybrid SOC model training
  final_tests/                       Primary/secondary evaluation scripts
  ekf_residuals/                     Residual profile discovery
  primary_transfer/                  Residual-shape transfer analysis
  robustness/                        Perturbation robustness checks
  secondary_transfer/                Adaptive Bayesian fusion

results/csv/                         Curated compact CSV result snapshots
docs/                                Reproducibility and source-selection notes
```

Large `.mat` workspaces, generated figures, MATLAB Live Scripts, and raw data are intentionally not tracked.

## Data

The raw battery dataset is available from the MATR repository:

https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

After downloading the three MATR batch files, place them in the MATLAB
working directory used for reproduction. The simplest option is the
repository root, i.e. the same folder that contains this `README.md`.

Rename the downloaded files to the names expected by `preprocessing.m`:

```text
MATR_batch_20170512.mat
MATR_batch_20170630.mat
MATR_batch_20180412.mat
```

These correspond to the MATR batch files dated 2017-05-12, 2017-06-30,
and 2018-04-12, respectively.


`battery_workspace_core.mat` is optional in the sense that it can be
regenerated from the raw MATR batch files by running the preprocessing
step. However, the authors recommend creating and keeping this file locally
after the first preprocessing run, so that the full preprocessing script
does not need to be executed every time the workflow is reproduced.

## Reproducing the Workflow

Open MATLAB, set a local working directory that contains or should receive the `.mat` artifacts, and run:

```matlab
workflow_work_dir = 'path/to/local/workdir';
run('code/matlab/run_reproduction_workflow.m')
```

The main generated model artifact is:

```text
hybrid_soc_model.mat
```

This MAT file is intentionally compact and stores only the trained hybrid residual model, its predictor list, and model metadata.

For detailed required files and execution order, see:

```text
docs/reproducibility_checklist.md
```

## Included Results

`results/csv/` contains compact CSV snapshots needed to inspect the numerical outputs without rerunning the full workflow.

## Citation

```bibtex
@article{Tsirkinidis2026,
  author  = {Tsirkinidis, Konstantinos and Savva, Christodoulos and Tournaviti, Maria and Michailidou, Alexandra V. and Vlachokostas, Christos},
  title   = {Hybrid modeling for state of charge estimation in {Battery Electric Vehicles}: {Balancing Accuracy, Efficiency, and Interpretability}},
  journal = {International Journal of Electrical Power and Energy Systems},
  year    = {2026},
  pages   = {111894},
  doi     = {10.1016/j.ijepes.2026.111894}
}
```

## Notes

`preprocessing.m` is preserved unchanged because it is externally sourced from

https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m
