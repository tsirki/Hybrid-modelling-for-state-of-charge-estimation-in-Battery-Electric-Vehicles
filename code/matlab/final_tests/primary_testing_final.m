%% =========================================================
% VARIANT C-LITE PRIMARY TESTING ONLY + Q_NOM SANITY CLAMP
%
% GOAL:
%   Test the already-trained Variant C-Lite model with:
%     - original Lite gate (same as previous better version)
%     - Q_nom sanity clamp only
%   on PRIMARY test batteries only
%
% IMPORTANT:
%   - No retraining
%   - Uses existing trained model: gpr_variantC_lite_model.mat
%   - Serial version
%
% REQUIRED IN WORKSPACE OR FILES:
%   t_all, I_all, V_all, Q_all
%   I_noise_std, V_noise_std
%   R0, R1, C1, R2, C2
%
% REQUIRED FILES:
%   gpr_variantC_lite_model.mat
%   Q_nom_init_first_cycle_all_batteries.mat
%
% OPTIONAL FILE:
%   fusion_full_model.mat
%
% OUTPUTS:
%   final_test_variantC_lite_qnomclamp_cycle_results_primary.csv
%   final_test_variantC_lite_qnomclamp_battery_summary_primary.csv
%   final_test_variantC_lite_qnomclamp_primary_outputs.mat
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

if exist('gpr_final_vC_lite', 'var') ~= 1 || exist('predictors', 'var') ~= 1
    load('gpr_variantC_lite_model.mat', 'gpr_final_vC_lite', 'predictors', 'model_info');
end

%% =========================================================
% SETTINGS
%% =========================================================
all_batteries = 1:numel(t_all);

% ---------------------------------------------------------
% EXPANDED TRAINING EXCLUSION SET FOR EVALUATION SPLIT
% ---------------------------------------------------------
added_train_batteries = [ ...
    42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82];

base_train_batteries = [];

if exist('train_batteries', 'var') == 1 && ~isempty(train_batteries)
    base_train_batteries = train_batteries(:);
elseif exist('model_info', 'var') == 1 && isstruct(model_info) && ...
       isfield(model_info, 'train_batteries') && ~isempty(model_info.train_batteries)
    base_train_batteries = model_info.train_batteries(:);
end

train_batteries_eval = unique([base_train_batteries; added_train_batteries(:)], 'stable')';

% ---------------------------------------------------------
% PRIMARY TEST SPLIT ONLY
%
% PRIMARY:
%   all batteries up to 84 that are NOT in training
%   and excluding battery 15
% ---------------------------------------------------------
primary_pool = 1:min(84, numel(t_all));
exclude_batteries = unique([train_batteries_eval(:); 15]);

primary_test_batteries = setdiff(primary_pool, exclude_batteries, 'stable');
test_batteries = primary_test_batteries(:)';

min_cycle_length = 30;
residual_clamp = 0.03;

% -------- original Lite trust scheduler --------
alpha_state_base = 0.50;
slope_gate_ref   = 0.08;
alpha_floor_mult = 0.25;
alpha_ema_mult   = 0.75;

% -------- Q_nom sanity clamp only --------
qnom_ratio_min = 0.60;           % lower bound for Q_nom / Q_nom_init
qnom_ratio_max = 1.05;           % upper bound
qnom_max_step_per_cycle = 0.03;  % max ratio step per cycle


% -------- filenames --------
save_csv = true;

cycle_csv_name   = 'final_test_variantC_lite_qnomclamp_cycle_results_primary.csv';
summary_csv_name = 'final_test_variantC_lite_qnomclamp_battery_summary_primary.csv';
outputs_mat_name = 'final_test_variantC_lite_qnomclamp_primary_outputs.mat';

%% =========================================================
% CHECKS
%% =========================================================
requiredVars = { ...
    't_all','I_all','V_all','Q_all', ...
    'Q_nom_init_per_battery', ...
    'I_noise_std','V_noise_std', ...
    'R0','R1','C1','R2','C2', ...
    'gpr_final_vC_lite','predictors'};

for k = 1:numel(requiredVars)
    if exist(requiredVars{k}, 'var') ~= 1
        error('Missing variable: %s', requiredVars{k});
    end
end

Q_nom_init_per_battery = Q_nom_init_per_battery(:);

fprintf('\n=========================================================\n');
fprintf('Testing previous Variant C-Lite + Q_nom sanity clamp\n');

fprintf('\nExpanded training exclusion set:\n');
 fprintf('%s\n', mat2str(train_batteries_eval));
fprintf('Number of excluded training batteries: %d\n', numel(train_batteries_eval));

fprintf('\nPrimary test batteries:\n');
fprintf('%s\n', mat2str(primary_test_batteries));
fprintf('Number of primary test batteries: %d\n', numel(primary_test_batteries));
fprintf('=========================================================\n');

%% =========================================================
% FULL TEST ON ALL CYCLES OF PRIMARY TEST BATTERIES
%% =========================================================
CycleResults = table();
BatterySummary = table();

for ib = 1:numel(test_batteries)
    battery_no = test_batteries(ib);

    if battery_no > numel(t_all) || isempty(t_all{battery_no}) || ...
       battery_no > numel(I_all) || isempty(I_all{battery_no}) || ...
       battery_no > numel(V_all) || isempty(V_all{battery_no}) || ...
       battery_no > numel(Q_all) || isempty(Q_all{battery_no})
        fprintf('Battery %d skipped: missing data\n', battery_no);
        continue;
    end

    if battery_no > numel(Q_nom_init_per_battery) || ...
       ~isfinite(Q_nom_init_per_battery(battery_no)) || ...
       Q_nom_init_per_battery(battery_no) <= 0
        fprintf('Battery %d skipped: invalid Q_nom_init_per_battery\n', battery_no);
        continue;
    end

    Q_nom_init_batt = Q_nom_init_per_battery(battery_no) * 3600;
    Q_nom = Q_nom_init_batt;

    num_cycles = numel(t_all{battery_no});
    if num_cycles < 1
        continue;
    end

    selected_cycles = unique([1, round((1 + num_cycles)/2), num_cycles]);
    fprintf('\nBattery %d | num_cycles = %d\n', battery_no, num_cycles);

    RowCell = cell(num_cycles,1);
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

    mean_alpha_lite_all = [];
    qnom_clamped_all = [];

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

            [Q_nom_next, qnom_was_clamped] = sanitize_qnom_next( ...
                Q_nom, Q_accumulated, Q_nom_init_batt, ...
                qnom_ratio_min, qnom_ratio_max, qnom_max_step_per_cycle);

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
                residual_pred(valid_te) = predict(gpr_final_vC_lite, Xtest(valid_te,:));
            end
            residual_pred = max(min(residual_pred, residual_clamp), -residual_clamp);

            % ---------------------------------------------------------
            % RAW GPR correction
            % ---------------------------------------------------------
            SOC_gpr = SOC_est + residual_pred;
            SOC_gpr = min(max(SOC_gpr, 0), 1);

            % ---------------------------------------------------------
            % ORIGINAL LITE correction
            % ---------------------------------------------------------
            alpha_lite = compute_lite_gate( ...
                soc_gate_alpha_ema, abs_dOCV_dSOC, ...
                alpha_state_base, slope_gate_ref, ...
                alpha_floor_mult, alpha_ema_mult);

            SOC_lite = SOC_est + alpha_lite .* residual_pred;
            SOC_lite = min(max(SOC_lite, 0), 1);

            % ---------------------------------------------------------
            % RMSEs
            % ---------------------------------------------------------
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

            mean_alpha_lite = mean(alpha_lite, 'omitnan');

            rmse_ekf_all = [rmse_ekf_all; rmse_ekf]; %#ok<AGROW>
            rmse_gpr_all = [rmse_gpr_all; rmse_gpr]; %#ok<AGROW>
            rmse_lite_all = [rmse_lite_all; rmse_lite]; %#ok<AGROW>

            rmse_ekf_late_all = [rmse_ekf_late_all; rmse_ekf_late]; %#ok<AGROW>
            rmse_gpr_late_all = [rmse_gpr_late_all; rmse_gpr_late]; %#ok<AGROW>
            rmse_lite_late_all = [rmse_lite_late_all; rmse_lite_late]; %#ok<AGROW>

            lite_improved_vs_ekf_all = [lite_improved_vs_ekf_all; is_lite_improved_vs_ekf]; %#ok<AGROW>
            lite_better_than_gpr_all = [lite_better_than_gpr_all; is_lite_better_than_gpr]; %#ok<AGROW>
            lite_worse_vs_ekf_all    = [lite_worse_vs_ekf_all; is_lite_worse_vs_ekf]; %#ok<AGROW>
            lite_worse_than_gpr_all  = [lite_worse_than_gpr_all; is_lite_worse_than_gpr]; %#ok<AGROW>

            mean_alpha_lite_all = [mean_alpha_lite_all; mean_alpha_lite]; %#ok<AGROW>
            qnom_clamped_all = [qnom_clamped_all; qnom_was_clamped]; %#ok<AGROW>

            is_selected_cycle = double(ismember(cycle_idx, selected_cycles));

            row_count = row_count + 1;
            RowCell{row_count} = table( ...
                battery_no, cycle_idx, ...
                Q_nom_cycle_start / Q_nom_init_batt, ...
                qnom_was_clamped, ...
                rmse_ekf, rmse_gpr, rmse_lite, ...
                delta_gpr_vs_ekf, delta_lite_vs_ekf, delta_lite_vs_gpr, ...
                rmse_ekf_late, rmse_gpr_late, rmse_lite_late, ...
                delta_gpr_vs_ekf_late, delta_lite_vs_ekf_late, delta_lite_vs_gpr_late, ...
                mean(abs(residual_pred), 'omitnan'), ...
                mean_alpha_lite, ...
                rmse_V, ...
                is_lite_improved_vs_ekf, ...
                is_lite_better_than_gpr, ...
                is_lite_worse_vs_ekf, ...
                is_lite_worse_than_gpr, ...
                is_selected_cycle, ...
                'VariableNames', { ...
                'battery_no','cycle_idx','qnom_start_frac', ...
                'qnom_was_clamped', ...
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

        catch err
            fprintf('Battery %d | cycle %d error: %s\n', battery_no, cycle_idx, err.message);
            continue;
        end
    end

    if row_count > 0
        CycleResults = [CycleResults; vertcat(RowCell{1:row_count})]; %#ok<AGROW>
    end

    if ~isempty(rmse_ekf_all)
        brow = table( ...
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
            mean(mean_alpha_lite_all,'omitnan'), ...
            sum(qnom_clamped_all,'omitnan'), ...
            'VariableNames', { ...
            'battery_no','num_cycles_evaluated', ...
            'mean_rmse_ekf','mean_rmse_gpr','mean_rmse_lite', ...
            'mean_rmse_ekf_late','mean_rmse_gpr_late','mean_rmse_lite_late', ...
            'num_lite_improved_vs_ekf', ...
            'num_lite_better_than_gpr', ...
            'num_lite_worse_vs_ekf', ...
            'num_lite_worse_than_gpr', ...
            'lite_improve_rate_vs_ekf', ...
            'lite_better_rate_vs_gpr', ...
            'mean_alpha_lite', ...
            'num_qnom_clamped_cycles'});
        BatterySummary = [BatterySummary; brow]; %#ok<AGROW>
    end
end

if ~isempty(CycleResults)
    CycleResults = sortrows(CycleResults, {'battery_no','cycle_idx'});
end

if ~isempty(BatterySummary)
    BatterySummary = sortrows(BatterySummary, 'battery_no');
end

%% =========================================================
% DISPLAY
%% =========================================================
disp(' ');
disp('==================== Q_NOM CLAMP TEST CYCLE RESULTS (HEAD) ====================');
if ~isempty(CycleResults)
    disp(CycleResults(1:min(20,height(CycleResults)), :));
else
    disp('CycleResults is empty.');
end

disp(' ');
disp('==================== Q_NOM CLAMP TEST BATTERY SUMMARY ====================');
if ~isempty(BatterySummary)
    disp(BatterySummary);
else
    disp('BatterySummary is empty.');
end

if ~isempty(BatterySummary)
    fprintf('\n---------------- PRIMARY TESTING ----------------\n');
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
    fprintf('Total Lite worse than EKF      = %d\n', ...
        sum(BatterySummary.num_lite_worse_vs_ekf,'omitnan'));
    fprintf('Total Q_nom clamped cycles     = %d\n', ...
        sum(BatterySummary.num_qnom_clamped_cycles,'omitnan'));
end

%% =========================================================
% SAVE CSV
%% =========================================================
if save_csv
    if ~isempty(CycleResults)
        writetable(CycleResults, cycle_csv_name);
    else
        writetable(table(), cycle_csv_name);
    end

    if ~isempty(BatterySummary)
        writetable(BatterySummary, summary_csv_name);
    else
        writetable(table(), summary_csv_name);
    end

    fprintf('\nSaved:\n');
    fprintf('  %s\n', cycle_csv_name);
    fprintf('  %s\n', summary_csv_name);
end

%% =========================================================
% SAVE MAT OUTPUTS
%% =========================================================
save(outputs_mat_name, ...
    'CycleResults', ...
    'BatterySummary', ...
    'predictors', ...
    'test_batteries', ...
    'primary_test_batteries', ...
    'train_batteries_eval', ...
    'exclude_batteries', ...
    'alpha_state_base', ...
    'slope_gate_ref', ...
    'alpha_floor_mult', ...
    'alpha_ema_mult', ...
    'qnom_ratio_min', ...
    'qnom_ratio_max', ...
    'qnom_max_step_per_cycle');

fprintf('\nSaved outputs MAT:\n');
fprintf('  %s\n', outputs_mat_name);

%% =========================================================
% LOCAL FUNCTIONS
%% =========================================================
function y = causalEMA(x, alpha)
    x = x(:);
    y = zeros(size(x));
    if isempty(x), return; end
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

function [Q_nom_next, was_clamped] = sanitize_qnom_next( ...
    Q_nom_current, Q_accumulated, Q_nom_init_batt, ...
    qnom_ratio_min, qnom_ratio_max, qnom_max_step_per_cycle)

    was_clamped = false;
    Q_nom_next = Q_nom_current;

    if ~isfinite(Q_nom_current) || Q_nom_current <= 0 || ...
       ~isfinite(Q_nom_init_batt) || Q_nom_init_batt <= 0
        return;
    end

    if ~isfinite(Q_accumulated) || Q_accumulated <= 0
        return;
    end

    qnom_candidate = Q_accumulated / 2;
    if ~isfinite(qnom_candidate) || qnom_candidate <= 0
        return;
    end

    ratio_prev = Q_nom_current / Q_nom_init_batt;
    ratio_cand = qnom_candidate / Q_nom_init_batt;
    ratio_raw = ratio_cand;

    ratio_cand = min(max(ratio_cand, qnom_ratio_min), qnom_ratio_max);

    ratio_low  = max(qnom_ratio_min, ratio_prev - qnom_max_step_per_cycle);
    ratio_high = min(qnom_ratio_max, ratio_prev + qnom_max_step_per_cycle);

    ratio_next = min(max(ratio_cand, ratio_low), ratio_high);

    if abs(ratio_next - ratio_raw) > 1e-12
        was_clamped = true;
    end

    Q_nom_next = ratio_next * Q_nom_init_batt;
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
