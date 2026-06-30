# Hybrid Modeling for State-of-Charge Estimation in Battery Electric Vehicles

MATLAB code and curated numerical outputs accompanying:

> Tsirkinidis, K., Savva, C., Tournaviti, M., Michailidou, A. V., and Vlachokostas, C. *Hybrid Modeling for State-of-Charge Estimation in Battery Electric Vehicles: Balancing Accuracy, Efficiency, and Interpretability.* *International Journal of Electrical Power and Energy Systems*, 2026, Article 111894. https://doi.org/10.1016/j.ijepes.2026.111894

## Overview

This repository provides the MATLAB implementation and curated numerical result tables associated with the paper above.

The workflow combines:

* an Extended Kalman Filter (EKF) based on a two-RC Thevenin equivalent-circuit model;
* sparse Gaussian Process Regression (GPR) for residual correction; and
* adaptive Bayesian fusion for state-of-charge (SOC) estimation under varying battery conditions.

This repository is provided for research reproducibility and methodological inspection. It is not intended as production battery-management-system software.

## Paper

* **DOI:** https://doi.org/10.1016/j.ijepes.2026.111894
* **Journal:** *International Journal of Electrical Power and Energy Systems*
* **Year:** 2026
* **Article number:** 111894

## Requirements

The workflow requires:

* MATLAB R2020a or later;
* Statistics and Machine Learning Toolbox, required for Gaussian Process Regression through `fitrgp`.

Parallel execution is optional. Some scripts contain optional `parfor` settings, but the default workflow runs serially and does not require Parallel Computing Toolbox.

## Repository Structure

```text
code/matlab/
  run_reproduction_workflow.m        Ordered workflow entry point
  preprocessing/                     Raw-data preprocessing and compact configurations
  feature_selection/                 Deployable feature screening
  model_training/                    Hybrid EKF-GPR model training
  final_tests/                       Primary and secondary evaluation scripts
  ekf_residuals/                     Residual-profile discovery
  primary_transfer/                  Residual-shape transfer analysis
  robustness/                        Perturbation robustness checks
  secondary_transfer/                Adaptive Bayesian fusion

results/csv/                         Curated reference result snapshots
docs/                                Reproducibility and source-attribution notes
```

Large MATLAB workspace files, raw battery data, generated figures, MATLAB Live Scripts, and non-essential intermediate outputs are intentionally not tracked.

## Data Availability

The raw battery data are not distributed with this repository.

The workflow uses battery cycling data from the [MATR data repository](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204).

Download the required batch files and place them in the local MATLAB working directory used for reproduction. Rename the downloaded files as follows:

```text
MATR_batch_20170512.mat
MATR_batch_20170630.mat
MATR_batch_20180412.mat
```

These files correspond to the MATR batches dated 2017-05-12, 2017-06-30, and 2018-04-12, respectively.

The raw data remain subject to the terms, conditions, and citation requirements of the original MATR dataset.

## Reproducing the Workflow

### First run: starting from raw MATR data

Create a local working directory outside the repository. This directory must already exist and contain the three renamed MATR batch files.

In MATLAB, define the repository location and local working directory:

```matlab
repo_root = 'path/to/Hybrid-modelling-for-state-of-charge-estimation-in-Battery-Electric-Vehicles';

workflow_work_dir = 'path/to/local/matlab_working_directory';

workflow_cfg = struct('runPreprocessing', true);

run(fullfile(repo_root, 'code', 'matlab', 'run_reproduction_workflow.m'))
```

The preprocessing stage creates:

```text
battery_workspace_core.mat
```

This local workspace file contains the compact core variables required by the downstream workflow stages.

### Subsequent runs

After `battery_workspace_core.mat` has been created, preprocessing does not need to be repeated.

```matlab
repo_root = 'path/to/Hybrid-modelling-for-state-of-charge-estimation-in-Battery-Electric-Vehicles';

workflow_work_dir = 'path/to/local/matlab_working_directory';

workflow_cfg = struct();

run(fullfile(repo_root, 'code', 'matlab', 'run_reproduction_workflow.m'))
```

The workflow checks for required intermediate files before each stage. If a required file is missing, MATLAB reports the relevant filename and workflow stage.

For the detailed list of required local files and execution order, see:

```text
docs/reproducibility_checklist.md
```

## Workflow Outputs

The principal trained-model artifact is:

```text
hybrid_soc_model.mat
```

This compact MAT file stores:

* the trained sparse GPR residual model;
* the predictor list;
* model hyperparameters and metadata;
* training and test battery identifiers; and
* grouped battery-level cross-validation summary metrics.

It does not include the raw battery time-series data.

## Included Results

The directory:

```text
results/csv/
```

contains curated CSV snapshots of selected numerical outputs from the release workflow.

These files allow readers to inspect the principal numerical results without rerunning the complete MATLAB analysis. They should be treated as reference outputs for this repository version.

Minor numerical differences may occur across MATLAB releases, operating systems, or numerical-library versions. Comparisons should therefore focus on the reported metrics and output trends rather than binary-identical output files.

## Citation

If you use this repository, please cite the associated paper:

```bibtex
@article{Tsirkinidis2026,
  author  = {Tsirkinidis, Konstantinos and Savva, Christodoulos and
             Tournaviti, Maria and Michailidou, Alexandra V. and
             Vlachokostas, Christos},
  title   = {Hybrid Modeling for State-of-Charge Estimation in
             {Battery Electric Vehicles}: Balancing Accuracy, Efficiency,
             and Interpretability},
  journal = {International Journal of Electrical Power and Energy Systems},
  year    = {2026},
  pages   = {111894},
  doi     = {10.1016/j.ijepes.2026.111894}
}
```

## Data and Third-Party Code Attribution

The raw battery data are obtained separately from the MATR repository and are not redistributed here.

Parts of the data-loading and batch-combination logic in:

```text
code/matlab/preprocessing/preprocessing.m
```

were adapted from the [`LoadData.m` workflow](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m) released in connection with the work of Severson, Attia, and co-authors on battery cycle-life prediction.

The adapted code is retained solely to support reproducibility of the present workflow. See `NOTICE.md` for source attribution and third-party code information.

## License

Unless otherwise stated in `NOTICE.md`, the original code and documentation developed for this repository are released under the MIT License.

External datasets and third-party code are not relicensed by this repository and remain subject to their respective original terms, licenses, and attribution requirements.
