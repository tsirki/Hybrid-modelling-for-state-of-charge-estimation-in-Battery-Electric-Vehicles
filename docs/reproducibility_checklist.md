# Reproducibility Checklist

This repository is a curated code release. It does not include raw MATR
battery files, generated figure files, MATLAB Live Scripts, or large MATLAB
workspaces. Curated CSV result snapshots are included under `results/csv/`.

## Required Local Inputs

Place these files in the MATLAB working directory before running the full
workflow:

| File | Producer | Notes |
| --- | --- | --- |
| `battery_workspace_core.mat` | `preprocessing.m` plus the wrapper save step in `run_reproduction_workflow.m` | Must contain `t_all`, `I_all`, `V_all`, and `Q_all`. |
| `fusion_full_model.mat` | `code/matlab/preprocessing/create_fusion_model_config.m` | Contains EKF parameters, noise settings, and training battery indices. |
| `Q_nom_init_first_cycle_all_batteries.mat` | `code/matlab/preprocessing/extract_qnom_init_first_cycle.m` | Contains `Q_nom_init_per_battery`. |
| `hybrid_soc_model.mat` | `code/matlab/model_training/train_hybrid_soc_gpr_model.m` | Final sparse Hybrid EKF-GPR residual GPR model. |
| `residual_shape_model.mat` | `code/matlab/ekf_residuals/profile_discovery.m` | Required by `code/matlab/primary_transfer/overlap.m`. |

## Recommended Order

1. Open MATLAB in a clean working directory outside OneDrive.
2. Add the repository MATLAB folder to the path:
   ```matlab
   addpath(genpath('path/to/repo/code/matlab'))
   ```
3. Define the local working directory that contains or should receive the MAT
   files:
   ```matlab
   workflow_work_dir = 'path/to/local/matlab_outputs';
   ```
4. If raw MATR files are available, run:
   ```matlab
   workflow_cfg = struct('runPreprocessing', true);
   run('path/to/repo/code/matlab/run_reproduction_workflow.m')
   ```
5. If `battery_workspace_core.mat` already exists, run:
   ```matlab
   run('path/to/repo/code/matlab/run_reproduction_workflow.m')
   ```

## Reviewer Notes

- `preprocessing.m` is third-party/externally sourced code and is preserved
  unchanged in this release.
- Downstream scripts now load `battery_workspace_core.mat` when the required
  workspace variables are missing.
- Plot image export, per-battery CSV dumping, and non-essential evaluation MAT
  dumps are excluded from the release workflow to keep the repository focused.
- Scripts write generated CSV/MAT outputs to the current MATLAB working
  directory. The checked-in CSV snapshots under `results/csv/` are the curated
  reference outputs from the release workflow.
