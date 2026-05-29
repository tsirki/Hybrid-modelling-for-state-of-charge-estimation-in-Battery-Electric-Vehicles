# Hybrid EKF-GPR Battery SOC Estimation

This repository contains the core analysis code used for the manuscript:

> Hybrid Modeling for State of Charge Estimation in Battery Electric Vehicles: Balancing Accuracy, Efficiency, and Interpretability

The workflow combines an Extended Kalman Filter (EKF), sparse Gaussian Process Regression (GPR) residual learning, and adaptive Bayesian fusion for battery state-of-charge estimation.

## Repository Contents

- `code/matlab/preprocessing/`: raw-data preprocessing and compact workspace/config generation scripts.
- `code/matlab/feature_selection/`: final sparse residual-model feature workflow.
- `code/matlab/model_training/`: final Variant C-Lite GPR model training workflow.
- `code/matlab/final_tests/`: primary final-test scripts for the qnom-clamped EKF-GPR runs.
- `code/matlab/ekf_residuals/`: EKF residual profile discovery.
- `code/matlab/primary_transfer/`: primary-test transferability and residual-shape analysis.
- `code/matlab/robustness/`: robustness checks under perturbation scenarios.
- `code/matlab/secondary_transfer/`: secondary-test adaptive Bayesian fusion.

## Data Availability

https://data.matr.io/1/projects/5c48dd2bc625d700019f3204
Large MATLAB workspace files and raw battery datasets are not included in this repository. In the original analysis, files such as `battery_workspace_core.mat` were multi-GB workspaces and are intentionally excluded from GitHub.

Generated CSV outputs, plotting-only scripts, and figure files are intentionally excluded to keep the repository lightweight. Some core analysis scripts may still contain inline diagnostic plotting blocks from the original research workflow.

## Main Workflow

The recommended entry point is:

```matlab
run('code/matlab/run_reproduction_workflow.m')
```

Set `workflow_work_dir` to the local directory containing the required MAT
files before running it. See `docs/reproducibility_checklist.md` for the exact
file list and script order.

The scripts are organized in the order used for the paper:

1. Run raw-data preprocessing (`preprocessing.m`) to merge the MATR batches and build the MATLAB workspace variables used downstream.
2. Run `create_fusion_model_config.m` and `extract_qnom_init_first_cycle.m` to recreate the compact config and battery-specific initial-capacity MAT files.
3. Run the final sparse residual-model feature workflow.
4. Train the final Variant C-Lite GPR model with `train_variantC_lite_gpr_model.m`.
5. Run final EKF-GPR tests on primary and secondary battery groups.
6. Analyze EKF residual profile structure.
7. Evaluate transferability, robustness, and adaptive Bayesian fusion.

## Notes

`preprocessing.m` is preserved unchanged because it is externally sourced https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m.
