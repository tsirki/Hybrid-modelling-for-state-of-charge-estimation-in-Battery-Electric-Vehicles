# Source File Selection

This repository is a curated release, not a full dump of the original working directory.

The selected scripts were identified by matching the manuscript sections, generated output filenames, and modification dates near the final manuscript revision.

## Included Script Groups

| Repository path | Purpose |
| --- | --- |
| `code/matlab/run_reproduction_workflow.m` | Ordered wrapper for running the curated supplementary workflow from a clean MATLAB session. |
| `code/matlab/preprocessing/preprocessing.m` | Combines the MIT battery batches into `batch_combined` and builds the per-battery/per-cycle workspace containers: `t_all`, `I_all`, `V_all`, `T_all`, `Q_all`, `R0_all`, and `valid_cols_all`. |
| `code/matlab/preprocessing/create_fusion_model_config.m` | Recreates the compact `fusion_full_model.mat` configuration from explicit EKF/noise/training constants. The original source script for this MAT file was not found; all discovered scripts treated it as an input. |
| `code/matlab/preprocessing/extract_qnom_init_first_cycle.m` | Converted from `battery_nom_extract.mlx`; creates `Q_nom_init_first_cycle_all_batteries.mat` from the first cycle of each battery. |
| `code/matlab/feature_selection/stage1and2_for2GPRsv2.m` | Final feature-screening workflow for the two-GPR residual setup. |
| `code/matlab/model_training/train_variantC_lite_gpr_model.m` | Converted from the final `untitled.mlx` creator script; trains and saves `gpr_variantC_lite_model.mat`. |
| `code/matlab/final_tests/combined_qnomclamp_run.m` | Combined all/primary/secondary qnom-clamped run. |
| `code/matlab/final_tests/primary_testing_final.m` | Primary-test final EKF/GPR/qnom-clamp evaluation. |
| `code/matlab/ekf_residuals/profile_discovery.m` | EKF residual profile discovery and clustering. |
| `code/matlab/primary_transfer/overlap.m` | Primary transferability with residual-shape overlap analysis. |
| `code/matlab/robustness/check.m` | Robustness scenario evaluation. |
| `code/matlab/secondary_transfer/test_secondary_adaptive_bayes.m` | Secondary-test adaptive Bayesian fusion. |

## Derived Outputs

Generated CSV outputs are intentionally not tracked. The code can regenerate outputs such as:

- `fusion_full_model.mat` compact EKF/GPR configuration.
- `Q_nom_init_first_cycle_all_batteries.mat` battery-specific initial capacity values.
- `gpr_variantC_lite_model.mat` final Variant C-Lite GPR residual model.
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
