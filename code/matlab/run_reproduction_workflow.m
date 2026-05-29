%% RUN_REPRODUCTION_WORKFLOW
% Ordered entry point for reproducing the MATLAB analysis workflow.
%
% Run this script from a working directory that contains the required MAT
% files listed in docs/reproducibility_checklist.md. By default, raw-data
% preprocessing is disabled because the raw MATR batch files are not shipped
% with this repository.
%
% Optional override before running:
%   workflow_work_dir = 'C:\path\to\local\matlab_outputs';
%   workflow_cfg = struct('runPreprocessing', true, 'runSecondaryTransfer', false);
%   run('code/matlab/run_reproduction_workflow.m')
%
% Recommended use:
%   1. First run: set runPreprocessing=true to create battery_workspace_core.mat
%      from the raw MATR batch files.
%   2. Later runs: keep runPreprocessing=false and reuse the local
%      battery_workspace_core.mat cache. This avoids rerunning preprocessing
%      every time.

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

repo_root = fileparts(fileparts(script_dir));
if exist('workflow_work_dir', 'var') ~= 1 || isempty(workflow_work_dir)
    workflow_work_dir = repo_root;
end

if exist(workflow_work_dir, 'dir') ~= 7
    error('workflow_work_dir does not exist: %s', workflow_work_dir);
end

cd(workflow_work_dir);

cfg = struct();
cfg.runPreprocessing = false;  % set true to regenerate battery_workspace_core.mat
cfg.createFusionConfig = true;
cfg.extractQnomInit = true;
cfg.runFeatureSelection = true;
cfg.trainHybridSocModel = true;
cfg.runPrimaryFinalTest = true;
cfg.runCombinedFinalTest = true;
cfg.runProfileDiscovery = true;
cfg.runPrimaryTransfer = true;
cfg.runRobustness = true;
cfg.runSecondaryTransfer = true;

if exist('workflow_cfg', 'var') == 1
    cfg = merge_workflow_config(cfg, workflow_cfg);
end

fprintf('MATLAB reproduction workflow\n');
fprintf('Working directory: %s\n', pwd);
fprintf('Code directory:    %s\n\n', script_dir);

if cfg.runPreprocessing
    run_preprocessing_step(fullfile(script_dir, 'preprocessing', 'preprocessing.m'));
end

if cfg.createFusionConfig
    run_workflow_step('Create fusion model config', ...
        fullfile(script_dir, 'preprocessing', 'create_fusion_model_config.m'), {});
end

if cfg.extractQnomInit
    run_workflow_step('Extract first-cycle Q_nom_init', ...
        fullfile(script_dir, 'preprocessing', 'extract_qnom_init_first_cycle.m'), ...
        {'battery_workspace_core.mat'});
end

if cfg.runFeatureSelection
    run_workflow_step('Feature selection', ...
        fullfile(script_dir, 'feature_selection', 'stage1and2_for2GPRsv2.m'), ...
        {'battery_workspace_core.mat', 'fusion_full_model.mat', 'Q_nom_init_first_cycle_all_batteries.mat'});
end

if cfg.trainHybridSocModel
    run_workflow_step('Train Hybrid EKF-GPR GPR model', ...
        fullfile(script_dir, 'model_training', 'train_hybrid_soc_gpr_model.m'), ...
        {'battery_workspace_core.mat', 'fusion_full_model.mat', 'Q_nom_init_first_cycle_all_batteries.mat'});
end

if cfg.runPrimaryFinalTest
    run_workflow_step('Primary final test', ...
        fullfile(script_dir, 'final_tests', 'primary_testing_final.m'), ...
        {'battery_workspace_core.mat', 'fusion_full_model.mat', ...
         'Q_nom_init_first_cycle_all_batteries.mat', 'hybrid_soc_model.mat'});
end

if cfg.runCombinedFinalTest
    run_workflow_step('Combined primary/secondary final test', ...
        fullfile(script_dir, 'final_tests', 'combined_qnomclamp_run.m'), ...
        {'battery_workspace_core.mat', 'fusion_full_model.mat', ...
         'Q_nom_init_first_cycle_all_batteries.mat', 'hybrid_soc_model.mat'});
end

if cfg.runProfileDiscovery
    run_workflow_step('EKF residual profile discovery', ...
        fullfile(script_dir, 'ekf_residuals', 'profile_discovery.m'), ...
        {'battery_workspace_core.mat', 'Q_nom_init_first_cycle_all_batteries.mat'});
end

if cfg.runPrimaryTransfer
    run_workflow_step('Primary transfer and residual-shape overlap', ...
        fullfile(script_dir, 'primary_transfer', 'overlap.m'), ...
        {'battery_workspace_core.mat', 'fusion_full_model.mat', ...
         'Q_nom_init_first_cycle_all_batteries.mat', 'hybrid_soc_model.mat', ...
         'residual_shape_model.mat'});
end

if cfg.runRobustness
    run_workflow_step('Robustness checks', ...
        fullfile(script_dir, 'robustness', 'check.m'), ...
        {'battery_workspace_core.mat', 'fusion_full_model.mat', ...
         'Q_nom_init_first_cycle_all_batteries.mat', 'hybrid_soc_model.mat'});
end

if cfg.runSecondaryTransfer
    run_workflow_step('Secondary adaptive Bayesian fusion', ...
        fullfile(script_dir, 'secondary_transfer', 'test_secondary_adaptive_bayes.m'), ...
        {'battery_workspace_core.mat', 'fusion_full_model.mat', ...
         'Q_nom_init_first_cycle_all_batteries.mat', 'hybrid_soc_model.mat'});
end

fprintf('Workflow completed successfully.\n');

function cfg = merge_workflow_config(cfg, override)
names = fieldnames(override);
for i = 1:numel(names)
    if strcmp(names{i}, 'trainVariantCLiteModel')
        cfg.trainHybridSocModel = override.(names{i});
        continue;
    end

    if isfield(cfg, names{i})
        cfg.(names{i}) = override.(names{i});
    else
        error('Unknown workflow_cfg field: %s', names{i});
    end
end
end

function run_workflow_step(label, script_path, required_files)
fprintf('--- %s ---\n', label);

for i = 1:numel(required_files)
    if exist(required_files{i}, 'file') ~= 2
        error('Missing required file before "%s": %s', label, required_files{i});
    end
end

[script_folder, script_name] = fileparts(script_path);
addpath(script_folder);
work_dir = pwd;
cleanup_obj = onCleanup(@() cd(work_dir));
evalin('caller', script_name);
fprintf('Completed: %s\n\n', label);
end

function run_preprocessing_step(script_path)
fprintf('--- Raw preprocessing ---\n');

[script_folder, script_name] = fileparts(script_path);
addpath(script_folder);
work_dir = pwd;
cleanup_obj = onCleanup(@() cd(work_dir));
eval(script_name);

if exist('t_all', 'var') == 1 && exist('I_all', 'var') == 1 && ...
   exist('V_all', 'var') == 1 && exist('Q_all', 'var') == 1
    save('battery_workspace_core.mat', ...
        't_all', 'I_all', 'V_all', 'Q_all', '-v7.3');
    fprintf('Saved battery_workspace_core.mat from preprocessing workspace.\n');
    fprintf('Completed: Raw preprocessing\n\n');
else
    error('Preprocessing finished, but core variables t_all/I_all/V_all/Q_all were not found.');
end
end
