# Source File Selection

This repository is a curated release, not a full dump of the original working directory.

The selected scripts were identified by matching the manuscript sections, generated output filenames, and modification dates near the final manuscript revision.

## Included Script Groups

| Repository path | Purpose |
| --- | --- |
| `code/matlab/preprocessing/preprocess1.m` | Combines the MIT battery batches into `batch_combined` and performs early single-cycle preprocessing experiments. |
| `code/matlab/preprocessing/preprocess3_2.m` | Builds the per-battery/per-cycle workspace containers: `t_all`, `I_all`, `V_all`, `T_all`, `Q_all`, `R0_all`, and `valid_cols_all`. |
| `code/matlab/feature_selection/stage1and2_for2GPRsv2.m` | Final feature-screening workflow for the two-GPR residual setup. |
| `code/matlab/final_tests/primary_testing_final.m` | Primary-test final EKF/GPR/qnom-clamp evaluation. |
| `code/matlab/final_tests/combined_qnomclamp_run.m` | Combined all/primary/secondary qnom-clamped run. |
| `code/matlab/ekf_residuals/profile_discovery.m` | EKF residual profile discovery and clustering. |
| `code/matlab/primary_transfer/overlap.m` | Primary transferability with residual-shape overlap analysis. |
| `code/matlab/robustness/check.m` | Robustness scenario evaluation. |
| `code/matlab/secondary_transfer/test_secondary_adaptive_bayes.m` | Secondary-test adaptive Bayesian fusion. |

## Derived Outputs

Generated CSV outputs are intentionally not tracked. The code can regenerate outputs such as:

- EKF residual profile summaries.
- Primary qnom-clamped test outputs.
- Primary residual-shape overlap outputs.
- Robustness scenario outputs.
- Secondary adaptive Bayesian fusion outputs.

## Excluded Files

The following file classes were intentionally excluded:

- Large MATLAB workspace files such as `battery_workspace_core.mat`.
- MATLAB `.fig` files and large plotting packages.
- Raw battery datasets.
- Generated CSV result files.
- Generated figure files and plotting-only scripts.
- Per-battery secondary output folders.
- Python translations/experiments not used in the manuscript workflow.
- Timing/diagnostic scripts not required to reproduce the reported numerical results.
- Original MATLAB Live Script (`.mlx`) files after conversion to plain `.m` scripts.
- Older exploratory notebooks and duplicate `untitled` variants not matched to manuscript outputs.

This keeps the release focused on the manuscript workflow while avoiding large or ambiguous research artifacts.
