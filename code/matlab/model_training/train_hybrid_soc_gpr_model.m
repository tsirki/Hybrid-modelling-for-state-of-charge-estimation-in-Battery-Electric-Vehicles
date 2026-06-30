%% =========================================================
% Hybrid EKF-GPR + GROUPED 5-FOLD CV + PARFOR
%
% KEY IDEAS:
%   1) Replace anti-causal tau with causal progress_causal
%   2) Keep sparse GPR residual predictor
%   3) Use grouped 5-fold CV by battery (NOT row-wise CV)
%   4) Use parfor only where valid:
%        - over training batteries
%        - over CV folds
%        - over test batteries
%      but NOT over cycles because Q_nom is sequential per battery
%   5) Deployable mean-only correction:
%        SOC_lite = SOC_est + alpha_lite .* residual_pred
%
% REQUIRED IN WORKSPACE OR FILES:
%   t_all, I_all, V_all, Q_all
%   train_batteries
%   I_noise_std, V_noise_std
%   R0, R1, C1, R2, C2
%
% REQUIRED FILE:
%   Q_nom_init_first_cycle_all_batteries.mat
%
% OPTIONAL FILE:
%   fusion_full_model.mat
%
% OUTPUTS:
%   hybrid_soc_model.mat
%   hybrid_soc_grouped_battery_5fold_cv.csv
%   final_test_hybrid_soc_battery_summary_all.csv
%   final_test_hybrid_soc_battery_summary_primary.csv
%   final_test_hybrid_soc_battery_summary_secondary.csv
%   final_test_hybrid_soc_outputs.mat
%% =========================================================

clc; close all;
rng(42);

%% =========================================================
% CONDITIONAL LOADS
%% =========================================================
coreVars = {'t_all','I_all','V_all','Q_all'};
need_core_load = false;
for k = 1:numel(coreVars)
    if exist(coreVars{k}, 'var') ~= 1
        need_core_load = true;
        break;
    end
end

if need_core_load
    if exist('battery_workspace_core.mat', 'file') == 2
        load('battery_workspace_core.mat', 't_all','I_all','V_all','Q_all');
    else
        error('Missing core workspace variables and battery_workspace_core.mat was not found.');
    end
end

needed_fusion_vars = {'I_noise_std','V_noise_std','R0','R1','C1','R2','C2'};
need_fusion_load = false;
for k = 1:numel(needed_fusion_vars)
    if exist(needed_fusion_vars{k}, 'var') ~= 1
        need_fusion_load = true;
        break;
    end
end

if need_fusion_load
    load('fusion_full_model.mat');
end

if exist('Q_nom_init_per_battery', 'var') ~= 1
    load('Q_nom_init_first_cycle_all_batteries.mat', 'Q_nom_init_per_battery');
end

%% =========================================================
% SETTINGS
%% =========================================================
all_batteries = 1:numel(t_all);

% ---------------------------------------------------------
% EXPANDED TRAINING BATTERIES
% ---------------------------------------------------------
added_train_batteries = [ ...
    42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82];

train_batteries = unique([train_batteries(:); added_train_batteries(:)], 'stable')';

% ---------------------------------------------------------
% PRIMARY / SECONDARY TEST SPLIT
%
% PRIMARY:
%   all batteries up to 84 that are NOT in training
%   and excluding battery 15
%
% SECONDARY:
%   all remaining batteries not used for training
% ---------------------------------------------------------
primary_pool = 1:min(84, numel(t_all));
exclude_batteries = unique([train_batteries(:); 15]);

primary_test_batteries = setdiff(primary_pool, exclude_batteries, 'stable');
secondary_pool = setdiff(all_batteries, primary_pool, 'stable');
secondary_test_batteries = setdiff(secondary_pool, exclude_batteries, 'stable');

test_batteries = [primary_test_batteries(:); secondary_test_batteries(:)]';

min_cycle_length = 30;

% -------- sparse training design --------
cycle_stride_train = 10;
base_step = 80;
n_region1 = 8;
n_region2 = 12;
n_region3 = 16;
n_high_innov = 4;

region1 = [0.00 0.10];
region2 = [0.20 0.45];
region3 = [0.78 0.95];

% -------- remove flagged bad training cycles --------
bad_training_cycles = [ ...
    35 191
    25 391
    21 721
    39 641
    41 181
    33 141
    10   1
    16 221
    39   1
    25  21
    19 1011
    39 141
    31   1
    14 561
    33  61
     2   1
    31 481
    29 521
    41   1
     4   1];

% -------- Hybrid EKF-GPR predictors --------
predictors = { ...
    'soc_est', ...
    'progress_causal', ...
    'qnom_start_frac', ...
    'v_resid_abs_mean_so_far', ...
    'inv_abs_dOCV_dSOC_mean_so_far', ...
    'norm_innov_mean_so_far', ...
    'soc_gate_alpha_ema'};

% -------- GPR settings --------
gprKernel = 'ardsquaredexponential';
gprBasis  = 'constant';
gprFitMethod = 'sr';
gprPredictMethod = 'sr';
gprActiveSetSize = 15;
gprActiveSetMethod = 'sgma';
gprOptimizer = 'quasinewton';

% -------- residual clamp --------
residual_clamp = 0.03;

% -------- deployable trust scheduler --------
alpha_state_base = 0.50;
slope_gate_ref   = 0.08;
alpha_floor_mult = 0.25;
alpha_ema_mult   = 0.75;

% -------- CV + parallel settings --------
use_grouped_battery_cv = true;
nCVFolds = 5;

use_parallel_build = false;
use_parallel_cv    = false;
use_parallel_test  = false;
parallel_pool_profile = 'local';


% -------- filenames --------
save_csv = true;
save_cv_csv = true;

cv_csv_name      = 'hybrid_soc_grouped_battery_5fold_cv.csv';
summary_csv_name = 'final_test_hybrid_soc_battery_summary_all.csv';
primary_summary_csv_name = 'final_test_hybrid_soc_battery_summary_primary.csv';
secondary_summary_csv_name = 'final_test_hybrid_soc_battery_summary_secondary.csv';

model_mat_name   = 'hybrid_soc_model.mat';

%% =========================================================
% CHECKS
%% =========================================================
requiredVars = { ...
    't_all','I_all','V_all','Q_all', ...
    'train_batteries', ...
    'Q_nom_init_per_battery', ...
    'I_noise_std','V_noise_std', ...
    'R0','R1','C1','R2','C2'};

for k = 1:numel(requiredVars)
    if exist(requiredVars{k}, 'var') ~= 1
        error('Missing variable: %s', requiredVars{k});
    end
end

Q_nom_init_per_battery = Q_nom_init_per_battery(:);

fprintf('\n=========================================================\n');
fprintf('Expanded training batteries:\n');
fprintf('%s\n', mat2str(train_batteries));
fprintf('Number of training batteries: %d\n', numel(train_batteries));

fprintf('\nPrimary test batteries:\n');
fprintf('%s\n', mat2str(primary_test_batteries));
fprintf('Number of primary test batteries: %d\n', numel(primary_test_batteries));

fprintf('\nSecondary test batteries:\n');
fprintf('%s\n', mat2str(secondary_test_batteries));
fprintf('Number of secondary test batteries: %d\n', numel(secondary_test_batteries));
fprintf('=========================================================\n');

%% =========================================================
% PARALLEL POOL
%% =========================================================
parallel_ready = false;

if use_parallel_build || use_parallel_cv || use_parallel_test
    p = gcp('nocreate');
    if isempty(p)
        try
            parpool(parallel_pool_profile);
            parallel_ready = true;
        catch ME
            warning('Parallel pool could not start. Falling back to serial. Error: %s', ME.message);
            parallel_ready = false;
        end
    else
        parallel_ready = true;
    end
end

if ~parallel_ready
    use_parallel_build = false;
    use_parallel_cv    = false;
    use_parallel_test  = false;
end

%% =========================================================
% BUILD TRAINING TABLE (PARALLEL OVER BATTERIES)
%% =========================================================
fprintf('=========================================================\n');
fprintf('Building sparse training table for Hybrid EKF-GPR\n');
fprintf('Training batteries: %s\n', mat2str(train_batteries));
fprintf('Parallel build: %d\n', use_parallel_build);
fprintf('=========================================================\n');

TrainBlockCell = cell(numel(train_batteries),1);

if use_parallel_build
    parfor ib = 1:numel(train_batteries)
        battery_no = train_batteries(ib);

        TrainBlockCell{ib} = build_train_table_one_battery_lite( ...
            battery_no, ...
            t_all, I_all, V_all, Q_all, ...
            Q_nom_init_per_battery, ...
            I_noise_std, V_noise_std, ...
            R0, R1, C1, R2, C2, ...
            min_cycle_length, ...
            cycle_stride_train, ...
            base_step, n_region1, n_region2, n_region3, n_high_innov, ...
            region1, region2, region3, ...
            bad_training_cycles);
    end
else
    for ib = 1:numel(train_batteries)
        battery_no = train_batteries(ib);

        TrainBlockCell{ib} = build_train_table_one_battery_lite( ...
            battery_no, ...
            t_all, I_all, V_all, Q_all, ...
            Q_nom_init_per_battery, ...
            I_noise_std, V_noise_std, ...
            R0, R1, C1, R2, C2, ...
            min_cycle_length, ...
            cycle_stride_train, ...
            base_step, n_region1, n_region2, n_region3, n_high_innov, ...
            region1, region2, region3, ...
            bad_training_cycles);
    end
end

TrainBlockCell = TrainBlockCell(~cellfun(@isempty, TrainBlockCell));

if isempty(TrainBlockCell)
    error('Training table is empty.');
end

TrainTable = vertcat(TrainBlockCell{:});

fprintf('\nTraining samples collected: %d\n', height(TrainTable));
fprintf('Training batteries retained: %d\n', numel(unique(TrainTable.battery_no)));

%% =========================================================
% GROUPED 5-FOLD CV BY BATTERY
%% =========================================================
CVFoldSummary = table();

if use_grouped_battery_cv
    unique_train_bats = unique(TrainTable.battery_no(:))';
    nCVFolds_eff = min(nCVFolds, numel(unique_train_bats));

    if nCVFolds_eff < 2
        warning('Not enough unique training batteries for grouped CV. Skipping CV.');
    else
        fprintf('\n=========================================================\n');
        fprintf('Running grouped %d-fold CV by battery\n', nCVFolds_eff);
        fprintf('Parallel CV: %d\n', use_parallel_cv);
        fprintf('=========================================================\n');

        fold_battery_lists = make_grouped_battery_folds(unique_train_bats, nCVFolds_eff, 42);
        CVRowCell = cell(nCVFolds_eff,1);

        if use_parallel_cv
            parfor ff = 1:nCVFolds_eff
                rng(200000 + ff);

                val_batts = fold_battery_lists{ff};
                tr_mask = ~ismember(TrainTable.battery_no, val_batts);
                va_mask =  ismember(TrainTable.battery_no, val_batts);

                Ttr = TrainTable(tr_mask,:);
                Tva = TrainTable(va_mask,:);

                Xtr = Ttr{:, predictors};
                Ytr = Ttr.residual;

                Xva = Tva{:, predictors};
                Yva = Tva.residual;

                valid_tr = all(isfinite(Xtr),2) & isfinite(Ytr);
                valid_va = all(isfinite(Xva),2) & isfinite(Yva);

                Xtr = Xtr(valid_tr,:);
                Ytr = Ytr(valid_tr);

                Xva = Xva(valid_va,:);
                Yva = Yva(valid_va);
                batts_va = Tva.battery_no(valid_va);

                mdl_cv = fitrgp(Xtr, Ytr, ...
                    'KernelFunction', gprKernel, ...
                    'BasisFunction', gprBasis, ...
                    'Standardize', true, ...
                    'FitMethod', gprFitMethod, ...
                    'PredictMethod', gprPredictMethod, ...
                    'ActiveSetSize', min(gprActiveSetSize, size(Xtr,1)), ...
                    'ActiveSetMethod', gprActiveSetMethod, ...
                    'Optimizer', gprOptimizer);

                Yhat = predict(mdl_cv, Xva);
                Yhat = max(min(Yhat, residual_clamp), -residual_clamp);

                rmse_sample = sqrt(mean((Yhat - Yva).^2, 'omitnan'));
                mae_sample  = mean(abs(Yhat - Yva), 'omitnan');

                uva = unique(batts_va);
                batt_rmse = nan(numel(uva),1);
                batt_mae  = nan(numel(uva),1);

                for bb = 1:numel(uva)
                    idxb = batts_va == uva(bb);
                    batt_rmse(bb) = sqrt(mean((Yhat(idxb) - Yva(idxb)).^2, 'omitnan'));
                    batt_mae(bb)  = mean(abs(Yhat(idxb) - Yva(idxb)), 'omitnan');
                end

                CVRowCell{ff} = table( ...
                    ff, ...
                    numel(unique(Ttr.battery_no)), ...
                    numel(unique(Tva.battery_no)), ...
                    size(Xtr,1), ...
                    size(Xva,1), ...
                    rmse_sample, ...
                    mae_sample, ...
                    mean(batt_rmse,'omitnan'), ...
                    median(batt_rmse,'omitnan'), ...
                    mean(batt_mae,'omitnan'), ...
                    median(batt_mae,'omitnan'), ...
                    'VariableNames', { ...
                    'fold_idx', ...
                    'n_train_batteries', ...
                    'n_val_batteries', ...
                    'n_train_samples', ...
                    'n_val_samples', ...
                    'rmse_sample', ...
                    'mae_sample', ...
                    'mean_battery_rmse', ...
                    'median_battery_rmse', ...
                    'mean_battery_mae', ...
                    'median_battery_mae'});
            end
        else
            for ff = 1:nCVFolds_eff
                rng(200000 + ff);

                val_batts = fold_battery_lists{ff};
                tr_mask = ~ismember(TrainTable.battery_no, val_batts);
                va_mask =  ismember(TrainTable.battery_no, val_batts);

                Ttr = TrainTable(tr_mask,:);
                Tva = TrainTable(va_mask,:);

                Xtr = Ttr{:, predictors};
                Ytr = Ttr.residual;

                Xva = Tva{:, predictors};
                Yva = Tva.residual;

                valid_tr = all(isfinite(Xtr),2) & isfinite(Ytr);
                valid_va = all(isfinite(Xva),2) & isfinite(Yva);

                Xtr = Xtr(valid_tr,:);
                Ytr = Ytr(valid_tr);

                Xva = Xva(valid_va,:);
                Yva = Yva(valid_va);
                batts_va = Tva.battery_no(valid_va);

                mdl_cv = fitrgp(Xtr, Ytr, ...
                    'KernelFunction', gprKernel, ...
                    'BasisFunction', gprBasis, ...
                    'Standardize', true, ...
                    'FitMethod', gprFitMethod, ...
                    'PredictMethod', gprPredictMethod, ...
                    'ActiveSetSize', min(gprActiveSetSize, size(Xtr,1)), ...
                    'ActiveSetMethod', gprActiveSetMethod, ...
                    'Optimizer', gprOptimizer);

                Yhat = predict(mdl_cv, Xva);
                Yhat = max(min(Yhat, residual_clamp), -residual_clamp);

                rmse_sample = sqrt(mean((Yhat - Yva).^2, 'omitnan'));
                mae_sample  = mean(abs(Yhat - Yva), 'omitnan');

                uva = unique(batts_va);
                batt_rmse = nan(numel(uva),1);
                batt_mae  = nan(numel(uva),1);

                for bb = 1:numel(uva)
                    idxb = batts_va == uva(bb);
                    batt_rmse(bb) = sqrt(mean((Yhat(idxb) - Yva(idxb)).^2, 'omitnan'));
                    batt_mae(bb)  = mean(abs(Yhat(idxb) - Yva(idxb)), 'omitnan');
                end

                CVRowCell{ff} = table( ...
                    ff, ...
                    numel(unique(Ttr.battery_no)), ...
                    numel(unique(Tva.battery_no)), ...
                    size(Xtr,1), ...
                    size(Xva,1), ...
                    rmse_sample, ...
                    mae_sample, ...
                    mean(batt_rmse,'omitnan'), ...
                    median(batt_rmse,'omitnan'), ...
                    mean(batt_mae,'omitnan'), ...
                    median(batt_mae,'omitnan'), ...
                    'VariableNames', { ...
                    'fold_idx', ...
                    'n_train_batteries', ...
                    'n_val_batteries', ...
                    'n_train_samples', ...
                    'n_val_samples', ...
                    'rmse_sample', ...
                    'mae_sample', ...
                    'mean_battery_rmse', ...
                    'median_battery_rmse', ...
                    'mean_battery_mae', ...
                    'median_battery_mae'});
            end
        end

        CVFoldSummary = vertcat(CVRowCell{:});

        disp(' ');
        disp('==================== GROUPED BATTERY CV ====================');
        disp(CVFoldSummary);

        fprintf('\nGrouped CV summary:\n');
        fprintf('  Mean sample RMSE      = %.6f\n', mean(CVFoldSummary.rmse_sample, 'omitnan'));
        fprintf('  Mean sample MAE       = %.6f\n', mean(CVFoldSummary.mae_sample, 'omitnan'));
        fprintf('  Mean battery RMSE     = %.6f\n', mean(CVFoldSummary.mean_battery_rmse, 'omitnan'));
        fprintf('  Mean battery MAE      = %.6f\n', mean(CVFoldSummary.mean_battery_mae, 'omitnan'));

        if save_cv_csv
            writetable(CVFoldSummary, cv_csv_name);
            fprintf('Saved CV summary:\n  %s\n', cv_csv_name);
        end
    end
end

%% =========================================================
% TRAIN FINAL GPR ON ALL TRAINING DATA
%% =========================================================
Xtrain = TrainTable{:, predictors};
Ytrain = TrainTable.residual;

valid_tr = all(isfinite(Xtrain),2) & isfinite(Ytrain);
Xtrain = Xtrain(valid_tr,:);
Ytrain = Ytrain(valid_tr);

fprintf('\nTraining final sparse GPR with active set size = %d\n', gprActiveSetSize);

rng(42);
gpr_final_hybrid = fitrgp(Xtrain, Ytrain, ...
    'KernelFunction', gprKernel, ...
    'BasisFunction', gprBasis, ...
    'Standardize', true, ...
    'FitMethod', gprFitMethod, ...
    'PredictMethod', gprPredictMethod, ...
    'ActiveSetSize', min(gprActiveSetSize, size(Xtrain,1)), ...
    'ActiveSetMethod', gprActiveSetMethod, ...
    'Optimizer', gprOptimizer);

%% =========================================================
% SAVE TRAINED MODEL
%% =========================================================
model_info = struct();
model_info.model_name = 'hybrid_soc_model';
model_info.model_variable = 'gpr_final_hybrid';
model_info.variant = 'Hybrid EKF-GPR';
model_info.predictors = predictors;
model_info.kernel = gprKernel;
model_info.basis = gprBasis;
model_info.fit_method = gprFitMethod;
model_info.predict_method = gprPredictMethod;
model_info.active_set_size = gprActiveSetSize;
model_info.active_set_method = gprActiveSetMethod;
model_info.optimizer = gprOptimizer;
model_info.residual_clamp = residual_clamp;
model_info.train_batteries = train_batteries;
model_info.bad_training_cycles = bad_training_cycles;
model_info.training_samples = size(Xtrain,1);

model_info.alpha_state_base = alpha_state_base;
model_info.slope_gate_ref = slope_gate_ref;
model_info.alpha_floor_mult = alpha_floor_mult;
model_info.alpha_ema_mult = alpha_ema_mult;

model_info.use_grouped_battery_cv = use_grouped_battery_cv;
model_info.nCVFolds = nCVFolds;
model_info.primary_test_batteries = primary_test_batteries;
model_info.secondary_test_batteries = secondary_test_batteries;
model_info.exclude_batteries = exclude_batteries;

if ~isempty(CVFoldSummary)
    model_info.cv_mean_sample_rmse = mean(CVFoldSummary.rmse_sample, 'omitnan');
    model_info.cv_mean_sample_mae  = mean(CVFoldSummary.mae_sample, 'omitnan');
    model_info.cv_mean_battery_rmse = mean(CVFoldSummary.mean_battery_rmse, 'omitnan');
    model_info.cv_mean_battery_mae  = mean(CVFoldSummary.mean_battery_mae, 'omitnan');
else
    model_info.cv_mean_sample_rmse = NaN;
    model_info.cv_mean_sample_mae  = NaN;
    model_info.cv_mean_battery_rmse = NaN;
    model_info.cv_mean_battery_mae  = NaN;
end

save(model_mat_name, ...
    'gpr_final_hybrid', ...
    'predictors', ...
    'model_info');

fprintf('\nSaved trained model:\n');
fprintf('  %s\n', model_mat_name);

%% =========================================================
% FULL TEST ON ALL CYCLES OF ALL TEST BATTERIES
% (PARALLEL OVER BATTERIES)
%% =========================================================
fprintf('\n=========================================================\n');
fprintf('Full testing Hybrid EKF-GPR on all test batteries\n');
fprintf('Parallel test: %d\n', use_parallel_test);
fprintf('Primary test batteries: %s\n', mat2str(primary_test_batteries));
fprintf('Secondary test batteries: %s\n', mat2str(secondary_test_batteries));
fprintf('=========================================================\n');

CycleCell = cell(numel(test_batteries),1);
SummaryCell = cell(numel(test_batteries),1);

if use_parallel_test
    parfor ib = 1:numel(test_batteries)
        battery_no = test_batteries(ib);

        [CycleCell{ib}, SummaryCell{ib}] = evaluate_test_battery_hybrid_soc( ...
            battery_no, ...
            t_all, I_all, V_all, Q_all, ...
            Q_nom_init_per_battery, ...
            I_noise_std, V_noise_std, ...
            R0, R1, C1, R2, C2, ...
            min_cycle_length, ...
            gpr_final_hybrid, ...
            residual_clamp, ...
            alpha_state_base, slope_gate_ref, alpha_floor_mult, alpha_ema_mult);
    end
else
    for ib = 1:numel(test_batteries)
        battery_no = test_batteries(ib);

        [CycleCell{ib}, SummaryCell{ib}] = evaluate_test_battery_hybrid_soc( ...
            battery_no, ...
            t_all, I_all, V_all, Q_all, ...
            Q_nom_init_per_battery, ...
            I_noise_std, V_noise_std, ...
            R0, R1, C1, R2, C2, ...
            min_cycle_length, ...
            gpr_final_hybrid, ...
            residual_clamp, ...
            alpha_state_base, slope_gate_ref, alpha_floor_mult, alpha_ema_mult);
    end
end

CycleCell = CycleCell(~cellfun(@isempty, CycleCell));
SummaryCell = SummaryCell(~cellfun(@isempty, SummaryCell));

if ~isempty(CycleCell)
    CycleResults = vertcat(CycleCell{:});
    CycleResults = sortrows(CycleResults, {'battery_no','cycle_idx'});
else
    CycleResults = table();
end

if ~isempty(SummaryCell)
    BatterySummary = vertcat(SummaryCell{:});
    BatterySummary = sortrows(BatterySummary, 'battery_no');
else
    BatterySummary = table();
end

% ---------------------------------------------------------
% TAG PRIMARY / SECONDARY TEST GROUP
% ---------------------------------------------------------
if ~isempty(CycleResults)
    test_group = strings(height(CycleResults),1);
    test_group(:) = "secondary";
    test_group(ismember(CycleResults.battery_no, primary_test_batteries)) = "primary";
    CycleResults.test_group = categorical(test_group, {'primary','secondary'});
    CycleResults = movevars(CycleResults, 'test_group', 'Before', 'battery_no');
end

if ~isempty(BatterySummary)
    test_group = strings(height(BatterySummary),1);
    test_group(:) = "secondary";
    test_group(ismember(BatterySummary.battery_no, primary_test_batteries)) = "primary";
    BatterySummary.test_group = categorical(test_group, {'primary','secondary'});
    BatterySummary = movevars(BatterySummary, 'test_group', 'Before', 'battery_no');
end

PrimaryCycleResults = table();
SecondaryCycleResults = table();
PrimaryBatterySummary = table();
SecondaryBatterySummary = table();

if ~isempty(CycleResults)
    PrimaryCycleResults = CycleResults(CycleResults.test_group == "primary", :);
    SecondaryCycleResults = CycleResults(CycleResults.test_group == "secondary", :);
end

if ~isempty(BatterySummary)
    PrimaryBatterySummary = BatterySummary(BatterySummary.test_group == "primary", :);
    SecondaryBatterySummary = BatterySummary(BatterySummary.test_group == "secondary", :);
end

%% =========================================================
% DISPLAY
%% =========================================================
disp(' ');
disp('==================== FULL TEST CYCLE RESULTS (HEAD) ====================');
if ~isempty(CycleResults)
    disp(CycleResults(1:min(20,height(CycleResults)), :));
else
    disp('CycleResults is empty.');
end

disp(' ');
disp('==================== FULL TEST BATTERY SUMMARY ====================');
if ~isempty(BatterySummary)
    disp(BatterySummary);
else
    disp('BatterySummary is empty.');
end

if ~isempty(BatterySummary)
    fprintf('\n---------------- OVERALL ----------------\n');
    fprintf('Mean EKF RMSE         = %.6f\n', mean(BatterySummary.mean_rmse_ekf,'omitnan'));
    fprintf('Mean GPR RMSE         = %.6f\n', mean(BatterySummary.mean_rmse_gpr,'omitnan'));
    fprintf('Mean Lite RMSE        = %.6f\n', mean(BatterySummary.mean_rmse_lite,'omitnan'));

    fprintf('Mean EKF late RMSE    = %.6f\n', mean(BatterySummary.mean_rmse_ekf_late,'omitnan'));
    fprintf('Mean GPR late RMSE    = %.6f\n', mean(BatterySummary.mean_rmse_gpr_late,'omitnan'));
    fprintf('Mean Lite late RMSE   = %.6f\n', mean(BatterySummary.mean_rmse_lite_late,'omitnan'));

    fprintf('Mean Lite improve rate vs EKF  = %.2f %%\n', ...
        100 * mean(BatterySummary.lite_improve_rate_vs_ekf,'omitnan'));
    fprintf('Mean Lite better rate vs GPR   = %.2f %%\n', ...
        100 * mean(BatterySummary.lite_better_rate_vs_gpr,'omitnan'));
end

if ~isempty(PrimaryBatterySummary)
    fprintf('\n---------------- PRIMARY TESTING ----------------\n');
    fprintf('Mean EKF RMSE         = %.6f\n', mean(PrimaryBatterySummary.mean_rmse_ekf,'omitnan'));
    fprintf('Mean GPR RMSE         = %.6f\n', mean(PrimaryBatterySummary.mean_rmse_gpr,'omitnan'));
    fprintf('Mean Lite RMSE        = %.6f\n', mean(PrimaryBatterySummary.mean_rmse_lite,'omitnan'));

    fprintf('Mean EKF late RMSE    = %.6f\n', mean(PrimaryBatterySummary.mean_rmse_ekf_late,'omitnan'));
    fprintf('Mean GPR late RMSE    = %.6f\n', mean(PrimaryBatterySummary.mean_rmse_gpr_late,'omitnan'));
    fprintf('Mean Lite late RMSE   = %.6f\n', mean(PrimaryBatterySummary.mean_rmse_lite_late,'omitnan'));

    fprintf('Mean Lite improve rate vs EKF  = %.2f %%\n', ...
        100 * mean(PrimaryBatterySummary.lite_improve_rate_vs_ekf,'omitnan'));
    fprintf('Mean Lite better rate vs GPR   = %.2f %%\n', ...
        100 * mean(PrimaryBatterySummary.lite_better_rate_vs_gpr,'omitnan'));
end

if ~isempty(SecondaryBatterySummary)
    fprintf('\n---------------- SECONDARY TESTING ----------------\n');
    fprintf('Mean EKF RMSE         = %.6f\n', mean(SecondaryBatterySummary.mean_rmse_ekf,'omitnan'));
    fprintf('Mean GPR RMSE         = %.6f\n', mean(SecondaryBatterySummary.mean_rmse_gpr,'omitnan'));
    fprintf('Mean Lite RMSE        = %.6f\n', mean(SecondaryBatterySummary.mean_rmse_lite,'omitnan'));

    fprintf('Mean EKF late RMSE    = %.6f\n', mean(SecondaryBatterySummary.mean_rmse_ekf_late,'omitnan'));
    fprintf('Mean GPR late RMSE    = %.6f\n', mean(SecondaryBatterySummary.mean_rmse_gpr_late,'omitnan'));
    fprintf('Mean Lite late RMSE   = %.6f\n', mean(SecondaryBatterySummary.mean_rmse_lite_late,'omitnan'));

    fprintf('Mean Lite improve rate vs EKF  = %.2f %%\n', ...
        100 * mean(SecondaryBatterySummary.lite_improve_rate_vs_ekf,'omitnan'));
    fprintf('Mean Lite better rate vs GPR   = %.2f %%\n', ...
        100 * mean(SecondaryBatterySummary.lite_better_rate_vs_gpr,'omitnan'));
end

%% =========================================================
% SAVE CSV
%% =========================================================
if save_csv

    if ~isempty(BatterySummary)
        writetable(BatterySummary, summary_csv_name);
    else
        writetable(table(), summary_csv_name);
    end

    if ~isempty(PrimaryBatterySummary)
        writetable(PrimaryBatterySummary, primary_summary_csv_name);
    else
        writetable(table(), primary_summary_csv_name);
    end

    if ~isempty(SecondaryBatterySummary)
        writetable(SecondaryBatterySummary, secondary_summary_csv_name);
    else
        writetable(table(), secondary_summary_csv_name);
    end

    fprintf('\nSaved:\n');    fprintf('  %s\n', summary_csv_name);    fprintf('  %s\n', primary_summary_csv_name);    fprintf('  %s\n', secondary_summary_csv_name);
end

%% =========================================================
% MAT OUTPUTS
%% =========================================================
% Evaluation tables are exported as CSV only. The only model MAT saved by
% this workflow is hybrid_soc_model.mat above.

%% =========================================================
% LOCAL FUNCTIONS
%% =========================================================
function fold_battery_lists = make_grouped_battery_folds(unique_batteries, nFolds, seed)

    unique_batteries = unique_batteries(:)';
    rng(seed);

    perm = randperm(numel(unique_batteries));
    shuffled = unique_batteries(perm);

    fold_battery_lists = cell(nFolds,1);
    for i = 1:numel(shuffled)
        ff = mod(i-1, nFolds) + 1;
        fold_battery_lists{ff}(end+1) = shuffled(i); %#ok<AGROW>
    end
end

function TrainTableBatt = build_train_table_one_battery_lite( ...
    battery_no, ...
    t_all, I_all, V_all, Q_all, ...
    Q_nom_init_per_battery, ...
    I_noise_std, V_noise_std, ...
    R0, R1, C1, R2, C2, ...
    min_cycle_length, ...
    cycle_stride_train, ...
    base_step, n_region1, n_region2, n_region3, n_high_innov, ...
    region1, region2, region3, ...
    bad_training_cycles)

    rng(100000 + battery_no);

    TrainTableBatt = table();

    if battery_no > numel(t_all) || isempty(t_all{battery_no}) || ...
       battery_no > numel(I_all) || isempty(I_all{battery_no}) || ...
       battery_no > numel(V_all) || isempty(V_all{battery_no}) || ...
       battery_no > numel(Q_all) || isempty(Q_all{battery_no})
        return;
    end

    if battery_no > numel(Q_nom_init_per_battery) || ...
       ~isfinite(Q_nom_init_per_battery(battery_no)) || ...
       Q_nom_init_per_battery(battery_no) <= 0
        return;
    end

    Q_nom_init_batt = Q_nom_init_per_battery(battery_no) * 3600;
    Q_nom = Q_nom_init_batt;

    num_cycles = numel(t_all{battery_no});
    cycle_candidates = unique([1, 2:cycle_stride_train:num_cycles, num_cycles]);
    cycle_candidates = cycle_candidates(cycle_candidates >= 1 & cycle_candidates <= num_cycles);

    blockCell = cell(numel(cycle_candidates),1);
    blockCounter = 0;

    for cycle_idx = 1:num_cycles
        try
            if any(bad_training_cycles(:,1)==battery_no & bad_training_cycles(:,2)==cycle_idx)
                continue;
            end

            t_data = t_all{battery_no}{cycle_idx};
            I_data = I_all{battery_no}{cycle_idx};
            V_data = V_all{battery_no}{cycle_idx};
            Q_series = Q_all{battery_no}{cycle_idx};

            if isempty(t_data) || isempty(I_data) || isempty(V_data) || isempty(Q_series)
                continue;
            end

            t_data = t_data(:);
            I_data = I_data(:);
            V_data = V_data(:);
            Q_series = Q_series(:);

            min_len0 = min([numel(t_data), numel(I_data), numel(V_data), numel(Q_series)]);
            if min_len0 < min_cycle_length
                continue;
            end

            t_data = t_data(1:min_len0);
            I_data = I_data(1:min_len0);
            V_data = V_data(1:min_len0);
            Q_series = Q_series(1:min_len0);

            valid0 = isfinite(t_data) & isfinite(I_data) & isfinite(V_data) & isfinite(Q_series);
            t_data = t_data(valid0);
            I_data = I_data(valid0);
            V_data = V_data(valid0);
            Q_series = Q_series(valid0);

            if numel(t_data) < min_cycle_length
                continue;
            end

            I_data_noisy = I_data + I_noise_std * randn(size(I_data));
            V_data_noisy = V_data + V_noise_std * randn(size(V_data));

            Q_nom_cycle_start = Q_nom;

            [SOC_est, V_model, ~, Q_accumulated, debug] = ...
                ekf_thevenin_2RC_R0_adaptive_v2( ...
                    t_data, I_data_noisy, V_data_noisy, ...
                    Q_nom, R0, R1, C1, R2, C2, OCV_func_local());

            if isfinite(Q_accumulated/2) && (Q_accumulated/2) > 0
                Q_nom_next = Q_accumulated / 2;
            else
                Q_nom_next = Q_nom;
            end

            if ~ismember(cycle_idx, cycle_candidates)
                Q_nom = Q_nom_next;
                continue;
            end

            qmax = max(Q_series);
            if ~isfinite(qmax) || qmax <= 0
                Q_nom = Q_nom_next;
                continue;
            end

            SOC_true = Q_series / qmax;

            min_len = min([numel(t_data), numel(I_data), numel(V_data), numel(Q_series), ...
                           numel(SOC_true), numel(SOC_est), numel(V_model), ...
                           numel(debug.innovation), numel(debug.SOC_cc), ...
                           numel(debug.dOCV_dSOC), numel(debug.Rk_eff), ...
                           numel(debug.soc_gate_alpha)]);

            t = t_data(1:min_len);
            I_raw = I_data(1:min_len);
            V_raw = V_data(1:min_len);
            SOC_true = SOC_true(1:min_len);
            SOC_est = SOC_est(1:min_len);
            V_model = V_model(1:min_len);

            innovation = debug.innovation(1:min_len);
            dOCV_dSOC = debug.dOCV_dSOC(1:min_len);
            Rk_eff = debug.Rk_eff(1:min_len);
            soc_gate_alpha = debug.soc_gate_alpha(1:min_len);

            residual = SOC_true - SOC_est;
            V_resid = V_raw - V_model;

            abs_innov = abs(innovation);
            abs_dOCV_dSOC = abs(dOCV_dSOC);
            abs_v_resid = abs(V_resid);

            inv_abs_dOCV_dSOC = 1 ./ max(abs_dOCV_dSOC, 1e-8);
            norm_innov = abs_innov ./ sqrt(max(Rk_eff, 1e-12));
            soc_gate_alpha_ema = causalEMA(soc_gate_alpha, 0.05);

            v_resid_abs_mean_so_far = cummean_custom(abs_v_resid);
            inv_abs_dOCV_dSOC_mean_so_far = cummean_custom(inv_abs_dOCV_dSOC);
            norm_innov_mean_so_far = cummean_custom(norm_innov);

            progress_causal = build_progress_causal(t, I_raw, Q_nom_cycle_start);

            sample_idx = selectSparseIndices( ...
                progress_causal, abs_innov, base_step, ...
                n_region1, n_region2, n_region3, n_high_innov, ...
                region1, region2, region3);

            if isempty(sample_idx)
                Q_nom = Q_nom_next;
                continue;
            end

            blockCounter = blockCounter + 1;

            block = table();
            block.battery_no = repmat(battery_no, numel(sample_idx), 1);
            block.cycle_idx = repmat(cycle_idx, numel(sample_idx), 1);
            block.residual = residual(sample_idx);

            block.soc_est = SOC_est(sample_idx);
            block.progress_causal = progress_causal(sample_idx);
            block.qnom_start_frac = repmat(Q_nom_cycle_start / Q_nom_init_batt, numel(sample_idx), 1);
            block.v_resid_abs_mean_so_far = v_resid_abs_mean_so_far(sample_idx);
            block.inv_abs_dOCV_dSOC_mean_so_far = inv_abs_dOCV_dSOC_mean_so_far(sample_idx);
            block.norm_innov_mean_so_far = norm_innov_mean_so_far(sample_idx);
            block.soc_gate_alpha_ema = soc_gate_alpha_ema(sample_idx);

            blockCell{blockCounter} = block;

            Q_nom = Q_nom_next;

        catch
            continue;
        end
    end

    if blockCounter > 0
        TrainTableBatt = vertcat(blockCell{1:blockCounter});
    end
end

function [CycleResultsBatt, BatterySummaryBatt] = evaluate_test_battery_hybrid_soc( ...
    battery_no, ...
    t_all, I_all, V_all, Q_all, ...
    Q_nom_init_per_battery, ...
    I_noise_std, V_noise_std, ...
    R0, R1, C1, R2, C2, ...
    min_cycle_length, ...
    gpr_final_hybrid, ...
    residual_clamp, ...
    alpha_state_base, slope_gate_ref, alpha_floor_mult, alpha_ema_mult)

    rng(300000 + battery_no);

    CycleResultsBatt = table();
    BatterySummaryBatt = table();

    if battery_no > numel(t_all) || isempty(t_all{battery_no}) || ...
       battery_no > numel(I_all) || isempty(I_all{battery_no}) || ...
       battery_no > numel(V_all) || isempty(V_all{battery_no}) || ...
       battery_no > numel(Q_all) || isempty(Q_all{battery_no})
        return;
    end

    if battery_no > numel(Q_nom_init_per_battery) || ...
       ~isfinite(Q_nom_init_per_battery(battery_no)) || ...
       Q_nom_init_per_battery(battery_no) <= 0
        return;
    end

    Q_nom_init_batt = Q_nom_init_per_battery(battery_no) * 3600;
    Q_nom = Q_nom_init_batt;

    num_cycles = numel(t_all{battery_no});
    if num_cycles < 1
        return;
    end

    selected_cycles = unique([1, round((1 + num_cycles)/2), num_cycles]);

    rowCell = cell(num_cycles,1);
    row_count = 0;

    rmse_ekf_all = [];
    rmse_gpr_all = [];
    rmse_lite_all = [];

    rmse_ekf_late_all = [];
    rmse_gpr_late_all = [];
    rmse_lite_late_all = [];

    lite_improved_vs_ekf_all = [];
    lite_better_than_gpr_all = [];
    lite_worse_vs_ekf_all = [];
    lite_worse_than_gpr_all = [];

    for cycle_idx = 1:num_cycles
        try
            t_data = t_all{battery_no}{cycle_idx};
            I_data = I_all{battery_no}{cycle_idx};
            V_data = V_all{battery_no}{cycle_idx};
            Q_series = Q_all{battery_no}{cycle_idx};

            if isempty(t_data) || isempty(I_data) || isempty(V_data) || isempty(Q_series)
                continue;
            end

            t_data = t_data(:);
            I_data = I_data(:);
            V_data = V_data(:);
            Q_series = Q_series(:);

            min_len0 = min([numel(t_data), numel(I_data), numel(V_data), numel(Q_series)]);
            if min_len0 < min_cycle_length
                continue;
            end

            t_data = t_data(1:min_len0);
            I_data = I_data(1:min_len0);
            V_data = V_data(1:min_len0);
            Q_series = Q_series(1:min_len0);

            valid0 = isfinite(t_data) & isfinite(I_data) & isfinite(V_data) & isfinite(Q_series);
            t_data = t_data(valid0);
            I_data = I_data(valid0);
            V_data = V_data(valid0);
            Q_series = Q_series(valid0);

            if numel(t_data) < min_cycle_length
                continue;
            end

            I_data_noisy = I_data + I_noise_std * randn(size(I_data));
            V_data_noisy = V_data + V_noise_std * randn(size(V_data));

            Q_nom_cycle_start = Q_nom;

            [SOC_est, V_model, rmse_V, Q_accumulated, debug] = ...
                ekf_thevenin_2RC_R0_adaptive_v2( ...
                    t_data, I_data_noisy, V_data_noisy, ...
                    Q_nom, R0, R1, C1, R2, C2, OCV_func_local());

            if isfinite(Q_accumulated/2) && (Q_accumulated/2) > 0
                Q_nom_next = Q_accumulated / 2;
            else
                Q_nom_next = Q_nom;
            end

            qmax = max(Q_series);
            if ~isfinite(qmax) || qmax <= 0
                Q_nom = Q_nom_next;
                continue;
            end

            min_len = min([numel(t_data), numel(I_data), numel(V_data), numel(Q_series), ...
                           numel(SOC_est), numel(V_model), ...
                           numel(debug.innovation), numel(debug.dOCV_dSOC), ...
                           numel(debug.Rk_eff), numel(debug.soc_gate_alpha)]);

            t = t_data(1:min_len);
            I_raw = I_data(1:min_len);
            V_raw = V_data(1:min_len);
            SOC_true = Q_series(1:min_len) / qmax;
            SOC_est = SOC_est(1:min_len);
            V_model = V_model(1:min_len);

            innovation = debug.innovation(1:min_len);
            dOCV_dSOC = debug.dOCV_dSOC(1:min_len);
            Rk_eff = debug.Rk_eff(1:min_len);
            soc_gate_alpha = debug.soc_gate_alpha(1:min_len);

            V_resid = V_raw - V_model;

            abs_innov = abs(innovation);
            abs_dOCV_dSOC = abs(dOCV_dSOC);
            abs_v_resid = abs(V_resid);

            inv_abs_dOCV_dSOC = 1 ./ max(abs_dOCV_dSOC, 1e-8);
            norm_innov = abs_innov ./ sqrt(max(Rk_eff, 1e-12));
            soc_gate_alpha_ema = causalEMA(soc_gate_alpha, 0.05);

            v_resid_abs_mean_so_far = cummean_custom(abs_v_resid);
            inv_abs_dOCV_dSOC_mean_so_far = cummean_custom(inv_abs_dOCV_dSOC);
            norm_innov_mean_so_far = cummean_custom(norm_innov);

            progress_causal = build_progress_causal(t, I_raw, Q_nom_cycle_start);

            Xtest = [ ...
                SOC_est, ...
                progress_causal, ...
                repmat(Q_nom_cycle_start / Q_nom_init_batt, numel(progress_causal), 1), ...
                v_resid_abs_mean_so_far, ...
                inv_abs_dOCV_dSOC_mean_so_far, ...
                norm_innov_mean_so_far, ...
                soc_gate_alpha_ema];

            valid_te = all(isfinite(Xtest),2);

            residual_pred = nan(size(SOC_est));
            if any(valid_te)
                residual_pred(valid_te) = predict(gpr_final_hybrid, Xtest(valid_te,:));
            end

            residual_pred = max(min(residual_pred, residual_clamp), -residual_clamp);

            SOC_gpr = SOC_est + residual_pred;
            SOC_gpr = min(max(SOC_gpr, 0), 1);

            alpha_lite = compute_lite_gate( ...
                soc_gate_alpha_ema, abs_dOCV_dSOC, ...
                alpha_state_base, slope_gate_ref, ...
                alpha_floor_mult, alpha_ema_mult);

            SOC_lite = SOC_est + alpha_lite .* residual_pred;
            SOC_lite = min(max(SOC_lite, 0), 1);

            rmse_ekf  = sqrt(mean((SOC_est  - SOC_true).^2, 'omitnan'));
            rmse_gpr  = sqrt(mean((SOC_gpr  - SOC_true).^2, 'omitnan'));
            rmse_lite = sqrt(mean((SOC_lite - SOC_true).^2, 'omitnan'));

            lateMask = progress_causal >= 0.80;

            if any(lateMask)
                rmse_ekf_late  = sqrt(mean((SOC_est(lateMask)  - SOC_true(lateMask)).^2, 'omitnan'));
                rmse_gpr_late  = sqrt(mean((SOC_gpr(lateMask)  - SOC_true(lateMask)).^2, 'omitnan'));
                rmse_lite_late = sqrt(mean((SOC_lite(lateMask) - SOC_true(lateMask)).^2, 'omitnan'));
            else
                rmse_ekf_late  = NaN;
                rmse_gpr_late  = NaN;
                rmse_lite_late = NaN;
            end

            delta_gpr_vs_ekf   = rmse_gpr  - rmse_ekf;
            delta_lite_vs_ekf  = rmse_lite - rmse_ekf;
            delta_lite_vs_gpr  = rmse_lite - rmse_gpr;

            delta_gpr_vs_ekf_late  = rmse_gpr_late  - rmse_ekf_late;
            delta_lite_vs_ekf_late = rmse_lite_late - rmse_ekf_late;
            delta_lite_vs_gpr_late = rmse_lite_late - rmse_gpr_late;

            is_lite_improved_vs_ekf = double(rmse_lite < rmse_ekf);
            is_lite_better_than_gpr = double(rmse_lite < rmse_gpr);

            is_lite_worse_vs_ekf   = double(rmse_lite > rmse_ekf);
            is_lite_worse_than_gpr = double(rmse_lite > rmse_gpr);

            rmse_ekf_all = [rmse_ekf_all; rmse_ekf]; %#ok<AGROW>
            rmse_gpr_all = [rmse_gpr_all; rmse_gpr]; %#ok<AGROW>
            rmse_lite_all = [rmse_lite_all; rmse_lite]; %#ok<AGROW>

            rmse_ekf_late_all = [rmse_ekf_late_all; rmse_ekf_late]; %#ok<AGROW>
            rmse_gpr_late_all = [rmse_gpr_late_all; rmse_gpr_late]; %#ok<AGROW>
            rmse_lite_late_all = [rmse_lite_late_all; rmse_lite_late]; %#ok<AGROW>

            lite_improved_vs_ekf_all = [lite_improved_vs_ekf_all; is_lite_improved_vs_ekf]; %#ok<AGROW>
            lite_better_than_gpr_all = [lite_better_than_gpr_all; is_lite_better_than_gpr]; %#ok<AGROW>
            lite_worse_vs_ekf_all = [lite_worse_vs_ekf_all; is_lite_worse_vs_ekf]; %#ok<AGROW>
            lite_worse_than_gpr_all = [lite_worse_than_gpr_all; is_lite_worse_than_gpr]; %#ok<AGROW>

            is_selected_cycle = double(ismember(cycle_idx, selected_cycles));

            row_count = row_count + 1;
            rowCell{row_count} = table( ...
                battery_no, cycle_idx, ...
                Q_nom_cycle_start / Q_nom_init_batt, ...
                rmse_ekf, rmse_gpr, rmse_lite, ...
                delta_gpr_vs_ekf, delta_lite_vs_ekf, delta_lite_vs_gpr, ...
                rmse_ekf_late, rmse_gpr_late, rmse_lite_late, ...
                delta_gpr_vs_ekf_late, delta_lite_vs_ekf_late, delta_lite_vs_gpr_late, ...
                mean(abs(residual_pred), 'omitnan'), ...
                mean(alpha_lite, 'omitnan'), ...
                rmse_V, ...
                is_lite_improved_vs_ekf, ...
                is_lite_better_than_gpr, ...
                is_lite_worse_vs_ekf, ...
                is_lite_worse_than_gpr, ...
                is_selected_cycle, ...
                'VariableNames', { ...
                'battery_no','cycle_idx','qnom_start_frac', ...
                'rmse_ekf','rmse_gpr','rmse_lite', ...
                'delta_gpr_vs_ekf','delta_lite_vs_ekf','delta_lite_vs_gpr', ...
                'rmse_ekf_late','rmse_gpr_late','rmse_lite_late', ...
                'delta_gpr_vs_ekf_late','delta_lite_vs_ekf_late','delta_lite_vs_gpr_late', ...
                'mean_abs_residual_pred','mean_alpha_lite','rmse_v', ...
                'is_lite_improved_vs_ekf', ...
                'is_lite_better_than_gpr', ...
                'is_lite_worse_vs_ekf', ...
                'is_lite_worse_than_gpr', ...
                'is_selected_cycle'});

            Q_nom = Q_nom_next;

        catch
            continue;
        end
    end

    if row_count > 0
        CycleResultsBatt = vertcat(rowCell{1:row_count});
    end

    if ~isempty(rmse_ekf_all)
        BatterySummaryBatt = table( ...
            battery_no, ...
            numel(rmse_ekf_all), ...
            mean(rmse_ekf_all,'omitnan'), ...
            mean(rmse_gpr_all,'omitnan'), ...
            mean(rmse_lite_all,'omitnan'), ...
            mean(rmse_ekf_late_all,'omitnan'), ...
            mean(rmse_gpr_late_all,'omitnan'), ...
            mean(rmse_lite_late_all,'omitnan'), ...
            sum(lite_improved_vs_ekf_all,'omitnan'), ...
            sum(lite_better_than_gpr_all,'omitnan'), ...
            sum(lite_worse_vs_ekf_all,'omitnan'), ...
            sum(lite_worse_than_gpr_all,'omitnan'), ...
            mean(lite_improved_vs_ekf_all,'omitnan'), ...
            mean(lite_better_than_gpr_all,'omitnan'), ...
            'VariableNames', { ...
            'battery_no','num_cycles_evaluated', ...
            'mean_rmse_ekf','mean_rmse_gpr','mean_rmse_lite', ...
            'mean_rmse_ekf_late','mean_rmse_gpr_late','mean_rmse_lite_late', ...
            'num_lite_improved_vs_ekf', ...
            'num_lite_better_than_gpr', ...
            'num_lite_worse_vs_ekf', ...
            'num_lite_worse_than_gpr', ...
            'lite_improve_rate_vs_ekf', ...
            'lite_better_rate_vs_gpr'});
    end
end

function idx = selectSparseIndices(progress_causal, abs_innov, base_step, ...
    n_region1, n_region2, n_region3, n_high_innov, region1, region2, region3)

    N = numel(progress_causal);
    idx = [];

    idx_base = unique(round(linspace(1, N, max(2, ceil(N / base_step)))));
    idx = [idx; idx_base(:)];

    idx = [idx; sampleRegion(progress_causal, region1, n_region1)];
    idx = [idx; sampleRegion(progress_causal, region2, n_region2)];
    idx = [idx; sampleRegion(progress_causal, region3, n_region3)];

    valid = isfinite(abs_innov);
    if any(valid)
        [~, order] = sort(abs_innov(valid), 'descend');
        validIdx = find(valid);
        pick = validIdx(order(1:min(n_high_innov, numel(order))));
        idx = [idx; pick(:)];
    end

    idx = unique(idx);
    idx = idx(idx >= 1 & idx <= N);
end

function idx = sampleRegion(x, region, n_pick)
    mask = x >= region(1) & x <= region(2);
    loc = find(mask);

    if isempty(loc)
        idx = [];
        return;
    end

    if numel(loc) <= n_pick
        idx = loc;
    else
        idx = unique(round(linspace(loc(1), loc(end), n_pick)));
    end

    idx = idx(:);
end

function y = causalEMA(x, alpha)
    x = x(:);
    y = zeros(size(x));

    if isempty(x)
        return;
    end

    y(1) = x(1);
    for k = 2:numel(x)
        y(k) = alpha * x(k) + (1 - alpha) * y(k-1);
    end
end

function m = cummean_custom(x)
    x = x(:);
    m = zeros(size(x));
    csum = 0;
    cnt = 0;

    for i = 1:numel(x)
        if isfinite(x(i))
            csum = csum + x(i);
            cnt = cnt + 1;
        end

        if cnt > 0
            m(i) = csum / cnt;
        else
            m(i) = NaN;
        end
    end
end

function progress_causal = build_progress_causal(t, I_raw, Q_nom_cycle_start)
    t = t(:);
    I_raw = I_raw(:);

    N = numel(t);
    progress_causal = zeros(N,1);

    if N < 2 || ~isfinite(Q_nom_cycle_start) || Q_nom_cycle_start <= 0
        return;
    end

    dt_sec = diff(t) * 60;
    dt_sec(~isfinite(dt_sec) | dt_sec < 0) = 0;

    I_eff = 1.1 * I_raw(2:end);
    dQ_abs = abs(I_eff .* dt_sec);

    Q_abs_cum = [0; cumsum(dQ_abs)];

    denom = max(2 * Q_nom_cycle_start, 1e-9);
    progress_causal = Q_abs_cum / denom;

    progress_causal(~isfinite(progress_causal)) = 0;
    progress_causal = min(max(progress_causal, 0), 1);
end

function alpha_lite = compute_lite_gate( ...
    soc_gate_alpha_ema, abs_dOCV_dSOC, ...
    alpha_state_base, slope_gate_ref, ...
    alpha_floor_mult, alpha_ema_mult)

    soc_gate_alpha_ema = soc_gate_alpha_ema(:);
    abs_dOCV_dSOC = abs_dOCV_dSOC(:);

    slope_gate = min(max(abs_dOCV_dSOC ./ max(slope_gate_ref, 1e-12), 0), 1);

    trust_gate = alpha_state_base .* ...
        (alpha_floor_mult + alpha_ema_mult .* soc_gate_alpha_ema);

    alpha_lite = trust_gate .* slope_gate;
    alpha_lite(~isfinite(alpha_lite)) = 0;
    alpha_lite = min(max(alpha_lite, 0), 1);
end

function OCV_func = OCV_func_local()
    OCV_func = @(soc) interp1( ...
        [0 0.02 0.05 0.10 0.20 0.40 0.60 0.80 0.90 0.95 0.98 1.00], ...
        [2.00 2.75 3.05 3.18 3.24 3.27 3.29 3.31 3.33 3.35 3.39 3.60], ...
        soc, 'pchip', 'extrap');
end

function [SOC_est, V_model, rmse_V, Q_accumulated, debug] = ...
    ekf_thevenin_2RC_R0_adaptive_v2(t, I, V_meas, Q_nom, R0, R1, C1, R2, C2, OCV_func)

    N = length(t);
    Q_accumulated = 0;

    x = [0; 0; 0];
    P = diag([1e-4, 1e-4, 1e-4]);

    q_soc  = 1e-7;
    q_vrc1 = 1e-4;
    q_vrc2 = 1e-4;
    Qk = diag([q_soc, q_vrc1, q_vrc2]);

    Rk_base = 1e-5;
    y_soc_freeze = 0.0223;
    y_soc_full   = 0.035;
    R_min = 1e-5;
    R_max = 5e-2;
    lambda_R = 0.999;
    innov_var = Rk_base;
    slope_thresh = 0.08;
    slope_boost_factor = 6;

    SOC_est = zeros(N, 1);
    V_model = 2 * ones(N, 1);

    debug.innovation = zeros(N,1);
    debug.SOC_cc = zeros(N,1);
    debug.dOCV_dSOC = zeros(N,1);
    debug.Rk_eff = zeros(N,1);
    debug.soc_gate_alpha = zeros(N,1);

    for k = 2:N
        Ik = 1.1 * I(k);
        delta_t = (t(k) - t(k-1)) * 60;

        Q_accumulated = Q_accumulated + abs(Ik * delta_t);

        SOC_pred = x(1) + (Ik * delta_t) / Q_nom;
        SOC_pred = min(max(SOC_pred, 0), 1);

        a1 = exp(-delta_t / (R1 * C1));
        a2 = exp(-delta_t / (R2 * C2));

        Vrc1_pred = a1 * x(2) + R1 * (1 - a1) * Ik;
        Vrc2_pred = a2 * x(3) + R2 * (1 - a2) * Ik;

        x_pred = [SOC_pred; Vrc1_pred; Vrc2_pred];

        F = eye(3);
        F(2,2) = a1;
        F(3,3) = a2;
        P_pred = F * P * F' + Qk;

        V_ocv_pred = OCV_func(SOC_pred);
        V_pred = V_ocv_pred + R0 * Ik + Vrc1_pred + Vrc2_pred;

        dOCV_dSOC = numerical_dOCV_dSOC(SOC_pred, OCV_func);
        H = [dOCV_dSOC, 1, 1];

        y = V_meas(k) - V_pred;
        abs_y = abs(y);

        innov_var = lambda_R * innov_var + (1 - lambda_R) * (y^2);
        Rk_eff = min(max(innov_var, R_min), R_max);

        if abs(dOCV_dSOC) < slope_thresh
            Rk_eff = min(Rk_eff * slope_boost_factor, R_max);
        end

        S = H * P_pred * H' + Rk_eff;
        K = P_pred * H' / S;

        if abs_y <= y_soc_freeze
            soc_gate_alpha = 0.0;
        elseif abs_y >= y_soc_full
            soc_gate_alpha = 1.0;
        else
            soc_gate_alpha = (abs_y - y_soc_freeze) / (y_soc_full - y_soc_freeze);
        end

        dx = K * y;
        dx(1) = soc_gate_alpha * dx(1);

        x = x_pred + dx;
        x(1) = min(max(x(1), 0), 1);

        K_eff = K;
        K_eff(1) = soc_gate_alpha * K_eff(1);
        P = (eye(3) - K_eff * H) * P_pred;

        V_ocv_corr = OCV_func(x(1));
        SOC_est(k) = x(1);
        V_model(k) = min(max(V_ocv_corr + R0 * Ik + x(2) + x(3), 2.0), 3.6);

        debug.SOC_cc(k) = x_pred(1);
        debug.innovation(k) = y;
        debug.dOCV_dSOC(k) = dOCV_dSOC;
        debug.Rk_eff(k) = Rk_eff;
        debug.soc_gate_alpha(k) = soc_gate_alpha;
    end

    rmse_V = sqrt(mean((V_meas - V_model).^2, 'omitnan'));
end

function d = numerical_dOCV_dSOC(soc, OCV_func)
    delta = 1e-5;
    soc1 = max(0, min(1, soc - delta));
    soc2 = max(0, min(1, soc + delta));

    if abs(soc2 - soc1) < 1e-12
        d = 0;
    else
        d = (OCV_func(soc2) - OCV_func(soc1)) / (soc2 - soc1);
    end
end
