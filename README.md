# Hybrid EKF-GPR Battery SOC Estimation

This repository contains the core analysis code used for the manuscript:

> Hybrid Modeling for State of Charge Estimation in Battery Electric Vehicles: Balancing Accuracy, Efficiency, and Interpretability

The workflow combines an Extended Kalman Filter (EKF), sparse Gaussian Process Regression (GPR) residual learning, and adaptive Bayesian fusion for battery state-of-charge estimation.

## Repository Contents

- `code/matlab/preprocessing/`: raw-data preprocessing scripts.
- `code/matlab/feature_selection/`: final sparse residual-model feature workflow.
- `code/matlab/final_tests/`: primary final-test scripts for the qnom-clamped EKF-GPR runs.
- `code/matlab/ekf_residuals/`: EKF residual profile discovery.
- `code/matlab/primary_transfer/`: primary-test transferability and residual-shape analysis.
- `code/matlab/robustness/`: robustness checks under perturbation scenarios.
- `code/matlab/secondary_transfer/`: secondary-test adaptive Bayesian fusion.

## Data Availability

Large MATLAB workspace files and raw battery datasets are not included in this repository. In the original analysis, files such as `battery_workspace_core.mat` were multi-GB workspaces and are intentionally excluded from GitHub.

Generated CSV outputs, plotting-only scripts, and figure files are intentionally excluded to keep the repository lightweight. Some core analysis scripts may still contain inline diagnostic plotting blocks from the original research workflow.

## Main Workflow

The scripts are organized roughly in the order used for the paper:

1. Run raw-data preprocessing (`preprocessing.m`) to merge the MATR batches and build the MATLAB workspace variables used downstream.
2. Run the final sparse residual-model feature workflow.
3. Run final EKF-GPR tests on primary and secondary battery groups.
4. Analyze EKF residual profile structure.
5. Evaluate transferability, robustness, and adaptive Bayesian fusion.

## Notes

The original MATLAB Live Scripts (`.mlx`) were converted to plain MATLAB scripts (`.m`) for easier review, diffing, and reuse in GitHub.
