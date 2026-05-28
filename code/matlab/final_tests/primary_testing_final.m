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

% -------- plotting --------
make_plots = false;
plot_selected_cycles_only = true;
plot_only_batteries = [1 3 5 7 11];

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

            if make_plots
                do_plot = true;

                if plot_selected_cycles_only && ~ismember(cycle_idx, selected_cycles)
                    do_plot = false;
                end

                if ~isempty(plot_only_batteries) && ~ismember(battery_no, plot_only_batteries)
                    do_plot = false;
                end

                if do_plot
                    figure('Name', sprintf('Battery %d Cycle %d', battery_no, cycle_idx), 'Color', 'w');
                    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

                    nexttile;
                    plot(t, SOC_true, 'k-', 'LineWidth', 1.8, 'DisplayName', 'True SOC'); hold on;
                    plot(t, SOC_est, 'b--', 'LineWidth', 1.2, 'DisplayName', 'EKF');
                    plot(t, SOC_gpr, 'm-', 'LineWidth', 1.2, 'DisplayName', 'EKF + GPR');
                    plot(t, SOC_lite, 'r-', 'LineWidth', 1.4, 'DisplayName', 'EKF + Lite');
                    xlabel('Time');
                    ylabel('SOC');
                    title(sprintf('SOC | Batt %d Cycle %d', battery_no, cycle_idx));
                    legend('Location','best');
                    grid on;

                    nexttile;
                    plot(t, SOC_true - SOC_est, 'k-', 'LineWidth', 1.4, 'DisplayName', 'True residual'); hold on;
                    plot(t, residual_pred, 'r--', 'LineWidth', 1.3, 'DisplayName', 'Pred residual');
                    plot(t, alpha_lite, 'b-', 'LineWidth', 1.2, 'DisplayName', 'alpha lite');
                    yline(0,'k:');
                    xlabel('Time');
                    ylabel('Residual / Weight');
                    title('Residual prediction + Lite gate');
                    legend('Location','best');
                    grid on;

                    nexttile;
                    plot(t, progress_causal, 'k-', 'LineWidth', 1.2, 'DisplayName', 'progress causal'); hold on;
                    plot(t, norm_innov_mean_so_far, 'b-', 'LineWidth', 1.2, 'DisplayName', 'norm innov mean');
                    plot(t, v_resid_abs_mean_so_far, 'r-', 'LineWidth', 1.2, 'DisplayName', 'v resid mean');
                    xlabel('Time');
                    ylabel('Feature value');
                    title('Key cumulative features');
                    legend('Location','best');
                    grid on;

                    nexttile;
                    plot(t, V_raw, 'k--', 'LineWidth', 1.0, 'DisplayName', 'Measured V'); hold on;
                    plot(t, V_model, 'b-', 'LineWidth', 1.3, 'DisplayName', 'Modelled V');
                    xlabel('Time');
                    ylabel('Voltage [V]');
                    title(sprintf('Voltage RMSE = %.4f', rmse_V));
                    legend('Location','best');
                    grid on;
                end
            end

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
%%
%% =========================================================
% COUNT RAW GPR WORSE-THAN-EKF CYCLES
%% =========================================================
clc; clear;

T = readtable('final_test_variantC_lite_qnomclamp_cycle_results_primary.csv');

eps0 = 1e-12;

is_gpr_worse = T.delta_gpr_vs_ekf > eps0;
is_gpr_better = T.delta_gpr_vs_ekf < -eps0;
is_gpr_equal = abs(T.delta_gpr_vs_ekf) <= eps0;

is_gpr_worse_late = T.delta_gpr_vs_ekf_late > eps0;
is_gpr_better_late = T.delta_gpr_vs_ekf_late < -eps0;
is_gpr_equal_late = abs(T.delta_gpr_vs_ekf_late) <= eps0;

nTotal = height(T);

fprintf('\n=========================================================\n');
fprintf('RAW GPR vs EKF (overall cycle RMSE)\n');
fprintf('Total evaluated cycles              : %d\n', nTotal);
fprintf('GPR worse than EKF                  : %d\n', sum(is_gpr_worse));
fprintf('GPR better than EKF                 : %d\n', sum(is_gpr_better));
fprintf('GPR equal to EKF                    : %d\n', sum(is_gpr_equal));
fprintf('GPR worse rate                      : %.4f %%\n', 100*sum(is_gpr_worse)/nTotal);

fprintf('\nRAW GPR vs EKF (late-region RMSE)\n');
fprintf('GPR worse than EKF in late region   : %d\n', sum(is_gpr_worse_late,'omitnan'));
fprintf('GPR better than EKF in late region  : %d\n', sum(is_gpr_better_late,'omitnan'));
fprintf('GPR equal to EKF in late region     : %d\n', sum(is_gpr_equal_late,'omitnan'));
fprintf('GPR worse late-rate                 : %.4f %%\n', 100*sum(is_gpr_worse_late,'omitnan')/nTotal);
fprintf('=========================================================\n');

%% Per-battery breakdown
ub = unique(T.battery_no);
BatteryGPR = table();

for i = 1:numel(ub)
    b = ub(i);
    idx = T.battery_no == b;

    Tb = T(idx,:);

    row = table( ...
        b, ...
        height(Tb), ...
        sum(Tb.delta_gpr_vs_ekf > eps0), ...
        sum(Tb.delta_gpr_vs_ekf < -eps0), ...
        sum(abs(Tb.delta_gpr_vs_ekf) <= eps0), ...
        100*mean(Tb.delta_gpr_vs_ekf > eps0), ...
        max(Tb.delta_gpr_vs_ekf, [], 'omitnan'), ...
        mean(Tb.delta_gpr_vs_ekf, 'omitnan'), ...
        sum(Tb.delta_gpr_vs_ekf_late > eps0), ...
        100*mean(Tb.delta_gpr_vs_ekf_late > eps0, 'omitnan'), ...
        'VariableNames', { ...
        'battery_no','num_cycles', ...
        'num_gpr_worse_vs_ekf', ...
        'num_gpr_better_vs_ekf', ...
        'num_gpr_equal_vs_ekf', ...
        'gpr_worse_rate_pct', ...
        'worst_delta_gpr_vs_ekf', ...
        'mean_delta_gpr_vs_ekf', ...
        'num_gpr_worse_vs_ekf_late', ...
        'gpr_worse_rate_pct_late'});
    
    BatteryGPR = [BatteryGPR; row]; %#ok<AGROW>
end

BatteryGPR = sortrows(BatteryGPR, 'num_gpr_worse_vs_ekf', 'descend');

disp(' ');
disp('==================== PER-BATTERY RAW GPR FAILURE COUNTS ====================');
disp(BatteryGPR);

%% Show worst offending cycles
Tworst = sortrows(T, 'delta_gpr_vs_ekf', 'descend');

disp(' ');
disp('==================== TOP 20 WORST RAW-GPR CYCLES ====================');
disp(Tworst(1:min(20,height(Tworst)), ...
    {'battery_no','cycle_idx','rmse_ekf','rmse_gpr','delta_gpr_vs_ekf', ...
     'rmse_ekf_late','rmse_gpr_late','delta_gpr_vs_ekf_late','qnom_was_clamped'}));
%%
%% =========================================================
% PUBLICATION PLOTS: RAW GPR vs EKF ONLY
%
% GOAL:
%   Highlight ONLY the improvement of raw GPR over EKF
%   using the PRIMARY CSV results.
%
% INPUT:
%   final_test_variantC_lite_qnomclamp_cycle_results_primary.csv
%   final_test_variantC_lite_qnomclamp_battery_summary_primary.csv
%
% OUTPUT DIR:
%   publication_plots_gpr_vs_ekf
%
% FIGURES:
%   Fig1_deltaRMSE_distribution_GPR_vs_EKF
%   Fig2_batterywise_meanRMSE_GPR_vs_EKF
%   Fig3_first_middle_last_GPR_vs_EKF
%   Fig4_soh_dependence_GPR_vs_EKF
%   Fig5_worst_case_cycles_GPR_vs_EKF
%
% SUMMARY TABLE:
%   gpr_vs_ekf_failure_counts_per_battery.csv
%% =========================================================

%clc; clear; close all;

%% =========================================================
% SETTINGS
%% =========================================================
cycleFile   = 'final_test_variantC_lite_qnomclamp_cycle_results_primary.csv';
batteryFile = 'final_test_variantC_lite_qnomclamp_battery_summary_primary.csv';

outDir = 'publication_plots_gpr_vs_ekf';
if ~exist(outDir,'dir')
    mkdir(outDir);
end

save_png = true;
save_pdf = true;
eps0 = 1e-12;

% Publication style
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultAxesLineWidth', 1.0);
set(groot, 'defaultLineLineWidth', 1.6);

% Colors
cEKF    = [0.15 0.15 0.15];
cGPR    = [0.55 0.20 0.75];
cFirst  = [0.00 0.4470 0.7410];
cMiddle = [0.8500 0.3250 0.0980];
cLast   = [0.9290 0.6940 0.1250];

phaseNames = {'FIRST','MIDDLE','LAST'};
xPhase = 1:3;

%% =========================================================
% LOAD
%% =========================================================
if exist(cycleFile,'file') ~= 2
    error('Cycle CSV not found: %s', cycleFile);
end
if exist(batteryFile,'file') ~= 2
    error('Battery CSV not found: %s', batteryFile);
end

T = readtable(cycleFile);
B = readtable(batteryFile);

fprintf('Loaded cycle file:   %s\n', cycleFile);
fprintf('Loaded battery file: %s\n', batteryFile);

%% =========================================================
% EXACT COUNTS: GPR WORSE / BETTER THAN EKF
%% =========================================================
is_gpr_worse = T.delta_gpr_vs_ekf > eps0;
is_gpr_better = T.delta_gpr_vs_ekf < -eps0;
is_gpr_equal = abs(T.delta_gpr_vs_ekf) <= eps0;

is_gpr_worse_late = T.delta_gpr_vs_ekf_late > eps0;
is_gpr_better_late = T.delta_gpr_vs_ekf_late < -eps0;
is_gpr_equal_late = abs(T.delta_gpr_vs_ekf_late) <= eps0;

nTotal = height(T);

fprintf('\n=========================================================\n');
fprintf('RAW GPR vs EKF (overall RMSE)\n');
fprintf('Total evaluated cycles            : %d\n', nTotal);
fprintf('GPR better than EKF               : %d\n', sum(is_gpr_better));
fprintf('GPR worse than EKF                : %d\n', sum(is_gpr_worse));
fprintf('GPR equal to EKF                  : %d\n', sum(is_gpr_equal));
fprintf('GPR improvement rate              : %.4f %%\n', 100*mean(is_gpr_better));
fprintf('GPR failure rate                  : %.4f %%\n', 100*mean(is_gpr_worse));

fprintf('\nRAW GPR vs EKF (late-region RMSE)\n');
fprintf('GPR better than EKF in late region: %d\n', sum(is_gpr_better_late,'omitnan'));
fprintf('GPR worse than EKF in late region : %d\n', sum(is_gpr_worse_late,'omitnan'));
fprintf('GPR equal in late region          : %d\n', sum(is_gpr_equal_late,'omitnan'));
fprintf('Late improvement rate             : %.4f %%\n', 100*mean(is_gpr_better_late,'omitnan'));
fprintf('Late failure rate                 : %.4f %%\n', 100*mean(is_gpr_worse_late,'omitnan'));
fprintf('=========================================================\n');

%% =========================================================
% PER-BATTERY FAILURE TABLE
%% =========================================================
ub = unique(T.battery_no);
BatteryGPR = table();

for i = 1:numel(ub)
    b = ub(i);
    Tb = T(T.battery_no == b, :);

    row = table( ...
        b, ...
        height(Tb), ...
        sum(Tb.delta_gpr_vs_ekf < -eps0), ...
        sum(Tb.delta_gpr_vs_ekf > eps0), ...
        100*mean(Tb.delta_gpr_vs_ekf < -eps0), ...
        100*mean(Tb.delta_gpr_vs_ekf > eps0), ...
        mean(Tb.delta_gpr_vs_ekf, 'omitnan'), ...
        min(Tb.delta_gpr_vs_ekf, [], 'omitnan'), ...
        max(Tb.delta_gpr_vs_ekf, [], 'omitnan'), ...
        sum(Tb.delta_gpr_vs_ekf_late < -eps0), ...
        sum(Tb.delta_gpr_vs_ekf_late > eps0), ...
        100*mean(Tb.delta_gpr_vs_ekf_late < -eps0, 'omitnan'), ...
        100*mean(Tb.delta_gpr_vs_ekf_late > eps0, 'omitnan'), ...
        'VariableNames', { ...
        'battery_no','num_cycles', ...
        'num_gpr_better_vs_ekf', ...
        'num_gpr_worse_vs_ekf', ...
        'gpr_improve_rate_pct', ...
        'gpr_failure_rate_pct', ...
        'mean_delta_gpr_vs_ekf', ...
        'best_delta_gpr_vs_ekf', ...
        'worst_delta_gpr_vs_ekf', ...
        'num_gpr_better_vs_ekf_late', ...
        'num_gpr_worse_vs_ekf_late', ...
        'gpr_improve_rate_pct_late', ...
        'gpr_failure_rate_pct_late'});
    
    BatteryGPR = [BatteryGPR; row]; %#ok<AGROW>
end

BatteryGPR = sortrows(BatteryGPR, 'num_gpr_worse_vs_ekf', 'descend');
writetable(BatteryGPR, fullfile(outDir, 'gpr_vs_ekf_failure_counts_per_battery.csv'));

disp(' ');
disp('==================== GPR FAILURE COUNTS PER BATTERY ====================');
disp(BatteryGPR);

%% =========================================================
% FIGURE 1: CYCLE-LEVEL DELTA RMSE DISTRIBUTION
%% =========================================================
f1 = figure('Position',[80 80 1300 900], 'Color','w');
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% Histogram overall
nexttile; hold on;
histogram(T.delta_gpr_vs_ekf, 60, ...
    'FaceColor', cGPR, 'FaceAlpha', 0.75, 'EdgeColor', 'none');
xline(0,'k--','LineWidth',1.2);
xlabel('\DeltaRMSE = raw GPR - EKF');
ylabel('Cycle count');
title('Distribution of raw GPR improvement over EKF', 'FontWeight','bold');
grid on; box off;

% Histogram late
nexttile; hold on;
histogram(T.delta_gpr_vs_ekf_late, 60, ...
    'FaceColor', cGPR, 'FaceAlpha', 0.75, 'EdgeColor', 'none');
xline(0,'k--','LineWidth',1.2);
xlabel('\DeltaRMSE_{late} = raw GPR - EKF');
ylabel('Cycle count');
title('Late-region distribution of raw GPR improvement', 'FontWeight','bold');
grid on; box off;

% CDF overall
nexttile; hold on;
plot_cdf(T.delta_gpr_vs_ekf, cGPR);
xline(0,'k--','LineWidth',1.2);
xlabel('\DeltaRMSE = raw GPR - EKF');
ylabel('Empirical CDF');
title('CDF of cycle-wise improvement over EKF', 'FontWeight','bold');
grid on; box off;

% CDF late
nexttile; hold on;
plot_cdf(T.delta_gpr_vs_ekf_late, cGPR);
xline(0,'k--','LineWidth',1.2);
xlabel('\DeltaRMSE_{late} = raw GPR - EKF');
ylabel('Empirical CDF');
title('CDF of late-region improvement over EKF', 'FontWeight','bold');
grid on; box off;

sgtitle('Cycle-level improvement of raw GPR over EKF', ...
    'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f1, outDir, 'Fig1_deltaRMSE_distribution_GPR_vs_EKF', save_png, save_pdf);

%% =========================================================
% FIGURE 2: BATTERY-WISE MEAN RMSE
%% =========================================================
B.mean_delta_gpr_vs_ekf = B.mean_rmse_gpr - B.mean_rmse_ekf;
B.mean_delta_gpr_vs_ekf_late = B.mean_rmse_gpr_late - B.mean_rmse_ekf_late;

[~, ordE] = sort(B.mean_rmse_ekf, 'ascend');
Bs = B(ordE,:);

f2 = figure('Position',[80 80 1300 900], 'Color','w');
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% Mean RMSE by battery
nexttile; hold on;
plot(Bs.mean_rmse_ekf, '-o', ...
    'Color', cEKF, 'MarkerFaceColor', cEKF, 'MarkerSize', 4);
plot(Bs.mean_rmse_gpr, '-s', ...
    'Color', cGPR, 'MarkerFaceColor', cGPR, 'MarkerSize', 4);
xlabel('Batteries (sorted by mean EKF RMSE)');
ylabel('Mean RMSE');
title('Battery-wise mean RMSE: EKF vs raw GPR', 'FontWeight','bold');
legend({'EKF','Raw GPR'}, 'Location','best', 'Box','off');
grid on; box off;

% Mean late RMSE by battery
nexttile; hold on;
plot(Bs.mean_rmse_ekf_late, '-o', ...
    'Color', cEKF, 'MarkerFaceColor', cEKF, 'MarkerSize', 4);
plot(Bs.mean_rmse_gpr_late, '-s', ...
    'Color', cGPR, 'MarkerFaceColor', cGPR, 'MarkerSize', 4);
xlabel('Batteries (sorted by mean EKF RMSE)');
ylabel('Mean late-region RMSE');
title('Battery-wise mean late RMSE: EKF vs raw GPR', 'FontWeight','bold');
legend({'EKF','Raw GPR'}, 'Location','best', 'Box','off');
grid on; box off;

% Scatter overall
nexttile; hold on;
scatter(B.mean_rmse_ekf, B.mean_rmse_gpr, 56, ...
    'filled', 'MarkerFaceColor', cGPR, 'MarkerEdgeColor', [0.2 0.2 0.2]);
lims = axis_limits_two(B.mean_rmse_ekf, B.mean_rmse_gpr);
plot(lims, lims, 'k--', 'LineWidth', 1.2);
xlim(lims); ylim(lims);
xlabel('Battery-wise mean EKF RMSE');
ylabel('Battery-wise mean raw GPR RMSE');
title('Battery-level accuracy comparison', 'FontWeight','bold');
grid on; box off;

% Sorted delta
nexttile; hold on;
vals = sort(B.mean_delta_gpr_vs_ekf, 'ascend');
plot(vals, '-o', ...
    'Color', cGPR, 'MarkerFaceColor', 'w', 'MarkerSize', 5, 'LineWidth', 2.0);
yline(0,'k--','LineWidth',1.2);
xlabel('Batteries (sorted)');
ylabel('Mean \DeltaRMSE = raw GPR - EKF');
title('Battery-wise gain relative to EKF', 'FontWeight','bold');
grid on; box off;

sgtitle('Battery-level improvement of raw GPR over EKF', ...
    'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f2, outDir, 'Fig2_batterywise_meanRMSE_GPR_vs_EKF', save_png, save_pdf);

%% =========================================================
% FIGURE 3: FIRST / MIDDLE / LAST TRANSITION ANALYSIS
%% =========================================================
Tsel = build_selected_triplets(T, phaseNames);

[battList, M_rmse_ekf] = metric_matrix(Tsel, 'rmse_ekf', phaseNames);
[~,       M_rmse_gpr]  = metric_matrix(Tsel, 'rmse_gpr', phaseNames);
[~,       M_delta]     = metric_matrix(Tsel, 'delta_gpr_vs_ekf', phaseNames);
[~,       M_deltaLate] = metric_matrix(Tsel, 'delta_gpr_vs_ekf_late', phaseNames);

f3 = figure('Position',[80 80 1300 900], 'Color','w');
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% RMSE transition
nexttile; hold on;
plot_battery_lines(xPhase, M_rmse_gpr, [0.83 0.83 0.83], 0.9);
plot(xPhase, mean(M_rmse_ekf,'omitnan'), '-o', ...
    'Color', cEKF, 'MarkerFaceColor', cEKF, 'MarkerSize', 8, 'LineWidth', 2.3);
plot(xPhase, mean(M_rmse_gpr,'omitnan'), '-s', ...
    'Color', cGPR, 'MarkerFaceColor', cGPR, 'MarkerSize', 8, 'LineWidth', 2.5);
format_phase_axis(phaseNames);
ylabel('RMSE');
title('Selected-cycle RMSE transition', 'FontWeight','bold');
legend({'Battery-wise raw GPR','EKF mean','Raw GPR mean'}, ...
    'Location','best', 'Box','off');
grid on; box off;

% Delta vs EKF
nexttile; hold on;
plot_battery_lines(xPhase, M_delta, [0.83 0.83 0.83], 0.9);
yline(0,'k--','LineWidth',1.2);
plot(xPhase, mean(M_delta,'omitnan'), '-s', ...
    'Color', cGPR, 'MarkerFaceColor', cGPR, 'MarkerSize', 8, 'LineWidth', 2.5);
format_phase_axis(phaseNames);
ylabel('\DeltaRMSE = raw GPR - EKF');
title('Improvement over EKF across life', 'FontWeight','bold');
grid on; box off;

% Late delta vs EKF
nexttile; hold on;
plot_battery_lines(xPhase, M_deltaLate, [0.83 0.83 0.83], 0.9);
yline(0,'k--','LineWidth',1.2);
plot(xPhase, mean(M_deltaLate,'omitnan'), '-s', ...
    'Color', cGPR, 'MarkerFaceColor', cGPR, 'MarkerSize', 8, 'LineWidth', 2.5);
format_phase_axis(phaseNames);
ylabel('\DeltaRMSE_{late} = raw GPR - EKF');
title('Late-region improvement across life', 'FontWeight','bold');
grid on; box off;

% Improvement rate across FIRST/MIDDLE/LAST
improveRateFML = 100 * [ ...
    mean(M_delta(:,1) < -eps0, 'omitnan'), ...
    mean(M_delta(:,2) < -eps0, 'omitnan'), ...
    mean(M_delta(:,3) < -eps0, 'omitnan')];

nexttile; hold on;
bar(xPhase, improveRateFML, 0.55, 'FaceColor', cGPR, 'EdgeColor', 'none');
format_phase_axis(phaseNames);
ylabel('Improvement rate vs EKF [%]');
title('Fraction of batteries improved in selected cycles', 'FontWeight','bold');
ylim([0 100]);
grid on; box off;

sgtitle('Selected FIRST / MIDDLE / LAST cycle analysis', ...
    'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f3, outDir, 'Fig3_first_middle_last_GPR_vs_EKF', save_png, save_pdf);

%% =========================================================
% FIGURE 4: SOH DEPENDENCE
%% =========================================================
f4 = figure('Position',[80 80 1300 900], 'Color','w');
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% Selected-cycle SOH dependence
nexttile; hold on;
plot_phase_scatter(Tsel.qnom_start_frac, Tsel.delta_gpr_vs_ekf, Tsel.phase, ...
    phaseNames, cFirst, cMiddle, cLast);
yline(0,'k--','LineWidth',1.2);
xlabel('Q_{nom,start} / Q_{nom,init}');
ylabel('\DeltaRMSE = raw GPR - EKF');
title('Selected-cycle improvement vs SOH proxy', 'FontWeight','bold');
legend(phaseNames, 'Location','best', 'Box','off');
grid on; box off;

% All-cycle SOH dependence
nexttile; hold on;
scatter(T.qnom_start_frac, T.delta_gpr_vs_ekf, 18, ...
    'filled', 'MarkerFaceColor', cGPR, 'MarkerEdgeColor', 'none', ...
    'MarkerFaceAlpha', 0.35);
yline(0,'k--','LineWidth',1.2);
xlabel('Q_{nom,start} / Q_{nom,init}');
ylabel('\DeltaRMSE = raw GPR - EKF');
title('All-cycle improvement vs SOH proxy', 'FontWeight','bold');
grid on; box off;

% Clamp sensitivity
nexttile; hold on;
idxClamp = logical(T.qnom_was_clamped);
scatter(T.qnom_start_frac(~idxClamp), T.delta_gpr_vs_ekf(~idxClamp), 18, ...
    'filled', 'MarkerFaceColor', [0.68 0.68 0.68], 'MarkerEdgeColor', 'none', ...
    'MarkerFaceAlpha', 0.35);
scatter(T.qnom_start_frac(idxClamp), T.delta_gpr_vs_ekf(idxClamp), 42, ...
    'filled', 'MarkerFaceColor', [0.85 0.15 0.15], 'MarkerEdgeColor', [0.2 0.2 0.2], ...
    'MarkerFaceAlpha', 0.90);
yline(0,'k--','LineWidth',1.2);
xlabel('Q_{nom,start} / Q_{nom,init}');
ylabel('\DeltaRMSE = raw GPR - EKF');
title('Q_{nom}-clamped cycles vs non-clamped cycles', 'FontWeight','bold');
legend({'Not clamped','Q_{nom} clamped'}, 'Location','best', 'Box','off');
grid on; box off;

% Battery summary SOH-ish proxy using mean qnom from cycle table
BatterySOH = groupsummary(T, 'battery_no', 'mean', 'qnom_start_frac');
X = nan(height(B),1);
for i = 1:height(B)
    idx = BatterySOH.battery_no == B.battery_no(i);
    if any(idx)
        X(i) = BatterySOH.mean_qnom_start_frac(idx);
    end
end

nexttile; hold on;
scatter(X, B.mean_delta_gpr_vs_ekf, 60, ...
    'filled', 'MarkerFaceColor', cGPR, 'MarkerEdgeColor', [0.2 0.2 0.2]);
yline(0,'k--','LineWidth',1.2);
xlabel('Battery mean Q_{nom,start} / Q_{nom,init}');
ylabel('Battery mean \DeltaRMSE = raw GPR - EKF');
title('Battery-level gain vs SOH proxy', 'FontWeight','bold');
grid on; box off;

sgtitle('SOH dependence of raw GPR improvement over EKF', ...
    'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f4, outDir, 'Fig4_soh_dependence_GPR_vs_EKF', save_png, save_pdf);

%% =========================================================
% FIGURE 5: WORST-CASE CYCLES
%% =========================================================
Tworst = sortrows(T, 'delta_gpr_vs_ekf', 'descend');
nShow = min(20, height(Tworst));
Tw = Tworst(1:nShow,:);

labels = strings(nShow,1);
for i = 1:nShow
    labels(i) = sprintf('B%d-C%d', Tw.battery_no(i), Tw.cycle_idx(i));
end

f5 = figure('Position',[80 80 1450 520], 'Color','w');
bar(Tw.delta_gpr_vs_ekf, 'FaceColor', cGPR, 'EdgeColor', 'none');
hold on;
yline(0,'k--','LineWidth',1.2);
xticks(1:nShow);
xticklabels(labels);
xtickangle(45);
ylabel('\DeltaRMSE = raw GPR - EKF');
title('Worst-case cycles where raw GPR underperforms EKF', 'FontWeight','bold');
grid on; box off;

save_figure_multi(f5, outDir, 'Fig5_worst_case_cycles_GPR_vs_EKF', save_png, save_pdf);

%% =========================================================
% CONSOLE SUMMARY
%% =========================================================
fprintf('\n=========================================================\n');
fprintf('Publication plot summary: raw GPR vs EKF only\n');
fprintf('Evaluated cycles        : %d\n', height(T));
fprintf('Evaluated batteries     : %d\n', height(B));
fprintf('Mean EKF RMSE           : %.6f\n', mean(T.rmse_ekf, 'omitnan'));
fprintf('Mean raw GPR RMSE       : %.6f\n', mean(T.rmse_gpr, 'omitnan'));
fprintf('Mean delta GPR-EKF      : %.6f\n', mean(T.delta_gpr_vs_ekf, 'omitnan'));
fprintf('GPR better than EKF     : %d cycles\n', sum(is_gpr_better));
fprintf('GPR worse than EKF      : %d cycles\n', sum(is_gpr_worse));
fprintf('Output folder           : %s\n', outDir);
fprintf('=========================================================\n');

%% =========================================================
% LOCAL FUNCTIONS
%% =========================================================
function Tsel = build_selected_triplets(T, phaseNames)

    ub = unique(T.battery_no);
    keepRows = [];

    for i = 1:numel(ub)
        b = ub(i);
        Tb = T(T.battery_no == b, :);
        cyc = sort(unique(Tb.cycle_idx));

        if isempty(cyc)
            continue;
        elseif numel(cyc) == 1
            chosen = cyc(1);
        elseif numel(cyc) == 2
            chosen = [cyc(1); cyc(end)];
        else
            midPos = round((1 + numel(cyc))/2);
            chosen = [cyc(1); cyc(midPos); cyc(end)];
        end

        for j = 1:numel(chosen)
            idx = find(T.battery_no == b & T.cycle_idx == chosen(j), 1, 'first');
            if ~isempty(idx)
                keepRows(end+1,1) = idx; %#ok<AGROW>
            end
        end
    end

    Tsel = T(keepRows,:);
    Tsel = sortrows(Tsel, {'battery_no','cycle_idx'});

    phaseOut = strings(height(Tsel),1);
    ub2 = unique(Tsel.battery_no);

    for i = 1:numel(ub2)
        idx = find(Tsel.battery_no == ub2(i));

        if numel(idx) == 1
            phaseOut(idx) = "MIDDLE";
        elseif numel(idx) == 2
            phaseOut(idx(1)) = "FIRST";
            phaseOut(idx(2)) = "LAST";
        else
            phaseOut(idx(1)) = "FIRST";
            phaseOut(idx(2)) = "MIDDLE";
            phaseOut(idx(3)) = "LAST";
        end
    end

    Tsel.phase = categorical(phaseOut, phaseNames, 'Ordinal', true);
end

function [battList, M] = metric_matrix(Tsel, metricName, phaseNames)
    battList = unique(Tsel.battery_no);
    M = nan(numel(battList), numel(phaseNames));

    for i = 1:numel(battList)
        Tb = Tsel(Tsel.battery_no == battList(i), :);
        for p = 1:numel(phaseNames)
            idx = Tb.phase == phaseNames{p};
            if any(idx)
                M(i,p) = Tb.(metricName)(find(idx,1,'first'));
            end
        end
    end
end

function plot_battery_lines(x, M, cLine, lw)
    for i = 1:size(M,1)
        yi = M(i,:);
        if all(isnan(yi))
            continue;
        end
        plot(x, yi, '-o', ...
            'Color', cLine, ...
            'LineWidth', lw, ...
            'MarkerSize', 4, ...
            'MarkerFaceColor', 'w');
    end
end

function format_phase_axis(phaseNames)
    xlim([0.7 3.3]);
    xticks(1:3);
    xticklabels(phaseNames);
end

function plot_phase_scatter(x, y, phase, phaseNames, cFirst, cMiddle, cLast)
    colors = {cFirst, cMiddle, cLast};
    markers = {'o','s','d'};

    for p = 1:numel(phaseNames)
        idx = phase == phaseNames{p};
        scatter(x(idx), y(idx), 54, ...
            'Marker', markers{p}, ...
            'MarkerEdgeColor', colors{p}, ...
            'MarkerFaceColor', colors{p}, ...
            'MarkerFaceAlpha', 0.82, ...
            'LineWidth', 0.8);
    end
end

function lims = axis_limits_two(x, y)
    x = x(isfinite(x));
    y = y(isfinite(y));
    lo = min([x(:); y(:)]);
    hi = max([x(:); y(:)]);
    pad = 0.05*(hi-lo+eps);
    lims = [lo-pad, hi+pad];
end

function h = plot_cdf(x, c)
    x = x(isfinite(x));
    if isempty(x)
        h = gobjects(1);
        return;
    end
    [f, xi] = ecdf(x);
    h = plot(xi, f, 'Color', c, 'LineWidth', 2.2);
end

function save_figure_multi(figHandle, outDir, baseName, save_png, save_pdf)
    if save_png
        exportgraphics(figHandle, fullfile(outDir, [baseName '.png']), 'Resolution', 300);
    end
    if save_pdf
        exportgraphics(figHandle, fullfile(outDir, [baseName '.pdf']), 'ContentType', 'vector');
    end
end
%%
%% =========================================================
% SECTION 3.2 STYLE PLOTS: EKF vs RAW GPR ONLY
%
% GOAL:
%   Generate publication-style plots analogous to the examples:
%     1) Spaghetti residuals for FIRST / MIDDLE / LAST cycles
%     2) Landmark plots vs SOH proxy
%     3) Battery-wise transition plots
%     4) Absolute error CDFs
%
% IMPORTANT:
%   - PRIMARY TEST BATTERIES ONLY
%   - EKF vs RAW GPR ONLY
%   - No Lite model in plots
%   - Uses same q_nom clamp logic and same GPR feature pipeline
%
% OUTPUT FOLDER:
%   gpr_vs_ekf_section32_figures
%
% OPTIONAL MAT OUTPUTS:
%   GPRSelTable.mat
%   GPRTrajTable.mat
%% =========================================================

clc; close all;
rng(42);

%% =========================================================
% OPTIONAL LOADS
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
    if exist('battery_workspace_core.mat','file') == 2
        load('battery_workspace_core.mat');
    else
        error('Missing core workspace vars and battery_workspace_core.mat was not found.');
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
    if exist('fusion_full_model.mat','file') == 2
        load('fusion_full_model.mat');
    else
        error('Missing fusion vars and fusion_full_model.mat was not found.');
    end
end

if exist('Q_nom_init_per_battery', 'var') ~= 1
    if exist('Q_nom_init_first_cycle_all_batteries.mat','file') == 2
        load('Q_nom_init_first_cycle_all_batteries.mat', 'Q_nom_init_per_battery');
    else
        error('Q_nom_init_first_cycle_all_batteries.mat not found.');
    end
end

if exist('gpr_final_vC_lite', 'var') ~= 1 || exist('predictors', 'var') ~= 1
    if exist('gpr_variantC_lite_model.mat','file') == 2
        load('gpr_variantC_lite_model.mat', 'gpr_final_vC_lite', 'predictors', 'model_info');
    else
        error('gpr_variantC_lite_model.mat not found.');
    end
end

%% =========================================================
% SETTINGS
%% =========================================================
outDir = 'gpr_vs_ekf_section32_figures';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

save_png = true;
save_pdf = true;
save_mat = true;

min_cycle_length = 30;
residual_clamp = 0.03;

% Q_nom clamp
qnom_ratio_min = 0.60;
qnom_ratio_max = 1.05;
qnom_max_step_per_cycle = 0.03;

% Normalized plotting grid
nGrid = 201;
tauGrid = linspace(0,1,nGrid)';

% Landmark windows
late_window = [0.70 0.98];
mid_window  = [0.15 0.60];

% Publication style
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultAxesLineWidth', 1.0);
set(groot, 'defaultLineLineWidth', 1.5);

% Colors
cEKF   = [0.12 0.12 0.12];
cGPR   = [0.00 0.15 1.00];
cGrey  = [0.72 0.72 0.72];
cGrey2 = [0.84 0.84 0.84];

phaseNames = {'FIRST','MIDDLE','LAST'};
phaseColors = { ...
    [0.00 0.4470 0.7410], ...
    [0.8500 0.3250 0.0980], ...
    [0.9290 0.6940 0.1250]};
markers = {'o','s','d'};

%% =========================================================
% PRIMARY TEST SPLIT (MATCHES YOUR EVALUATION)
%% =========================================================
added_train_batteries = [ ...
    42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82];

base_train_batteries = [];

if exist('train_batteries', 'var') == 1 && ~isempty(train_batteries)
    base_train_batteries = train_batteries(:);
elseif exist('model_info', 'var') == 1 && isstruct(model_info) && ...
       isfield(model_info, 'train_batteries') && ~isempty(model_info.train_batteries)
    base_train_batteries = model_info.train_batteries(:);
else
    error('Could not recover training batteries from train_batteries or model_info.train_batteries.');
end

train_batteries_eval = unique([base_train_batteries; added_train_batteries(:)], 'stable')';

primary_pool = 1:min(84, numel(t_all));
exclude_batteries = unique([train_batteries_eval(:); 15]);
primary_test_batteries = setdiff(primary_pool, exclude_batteries, 'stable');

fprintf('\n=========================================================\n');
fprintf('PRIMARY TEST BATTERIES USED FOR PLOTTING:\n');
fprintf('%s\n', mat2str(primary_test_batteries));
fprintf('Count = %d\n', numel(primary_test_batteries));
fprintf('=========================================================\n');

%% =========================================================
% MAIN COLLECTION LOOP
%% =========================================================
SelRows = {};
TrajRows = {};

for ib = 1:numel(primary_test_batteries)

    battery_no = primary_test_batteries(ib);

    fprintf('\nBattery %d (%d/%d)\n', battery_no, ib, numel(primary_test_batteries));

    if battery_no > numel(t_all) || isempty(t_all{battery_no}) || ...
       battery_no > numel(I_all) || isempty(I_all{battery_no}) || ...
       battery_no > numel(V_all) || isempty(V_all{battery_no}) || ...
       battery_no > numel(Q_all) || isempty(Q_all{battery_no})
        fprintf('  -> Missing data. Skipping.\n');
        continue;
    end

    if battery_no > numel(Q_nom_init_per_battery) || ...
       ~isfinite(Q_nom_init_per_battery(battery_no)) || ...
       Q_nom_init_per_battery(battery_no) <= 0
        fprintf('  -> Invalid Q_nom_init_per_battery. Skipping.\n');
        continue;
    end

    % -----------------------------------------------------
    % Determine valid cycles first
    % -----------------------------------------------------
    nCycles = numel(t_all{battery_no});
    valid_cycle_idx = [];

    for cycle_idx = 1:nCycles
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
            if sum(valid0) < min_cycle_length
                continue;
            end

            valid_cycle_idx(end+1,1) = cycle_idx; %#ok<AGROW>

        catch
            continue;
        end
    end

    if isempty(valid_cycle_idx)
        fprintf('  -> No valid cycles. Skipping.\n');
        continue;
    end

    if numel(valid_cycle_idx) == 1
        selected_cycles = valid_cycle_idx(1);
        selected_phase_labels = {'MIDDLE'};
    elseif numel(valid_cycle_idx) == 2
        selected_cycles = [valid_cycle_idx(1); valid_cycle_idx(end)];
        selected_phase_labels = {'FIRST'; 'LAST'};
    else
        midPos = round((1 + numel(valid_cycle_idx))/2);
        selected_cycles = [valid_cycle_idx(1); valid_cycle_idx(midPos); valid_cycle_idx(end)];
        selected_phase_labels = {'FIRST'; 'MIDDLE'; 'LAST'};
    end

    fprintf('  Selected cycles: %s\n', mat2str(selected_cycles(:)'));

    Q_nom_init_batt = Q_nom_init_per_battery(battery_no) * 3600;
    Q_nom = Q_nom_init_batt;

    for cycle_idx = 1:nCycles
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

            % Match your evaluation logic
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

            if ~ismember(cycle_idx, selected_cycles)
                Q_nom = Q_nom_next;
                continue;
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

            tau_plot = build_tau_plot(t);
            progress_causal = build_progress_causal(t, I_raw, Q_nom_cycle_start);

            abs_innov = abs(innovation);
            abs_dOCV_dSOC = abs(dOCV_dSOC);
            abs_v_resid = abs(V_resid);

            inv_abs_dOCV_dSOC = 1 ./ max(abs_dOCV_dSOC, 1e-8);
            norm_innov = abs_innov ./ sqrt(max(Rk_eff, 1e-12));
            soc_gate_alpha_ema = causalEMA(soc_gate_alpha, 0.05);

            v_resid_abs_mean_so_far = cummean_custom(abs_v_resid);
            inv_abs_dOCV_dSOC_mean_so_far = cummean_custom(inv_abs_dOCV_dSOC);
            norm_innov_mean_so_far = cummean_custom(norm_innov);

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

            SOC_gpr = SOC_est + residual_pred;
            SOC_gpr = min(max(SOC_gpr, 0), 1);

            resid_ekf = SOC_true - SOC_est;
            resid_gpr = SOC_true - SOC_gpr;

            rmse_ekf = sqrt(mean((SOC_est - SOC_true).^2, 'omitnan'));
            rmse_gpr = sqrt(mean((SOC_gpr - SOC_true).^2, 'omitnan'));

            abs_err_ekf = abs(resid_ekf);
            abs_err_gpr = abs(resid_gpr);

            res_energy_ekf = trapz(tau_plot, resid_ekf.^2);
            res_energy_gpr = trapz(tau_plot, resid_gpr.^2);

            [late_peak_ekf, late_t_ekf] = extract_peak_feature(tau_plot, resid_ekf, late_window, 'max');
            [late_peak_gpr, late_t_gpr] = extract_peak_feature(tau_plot, resid_gpr, late_window, 'max');

            [mid_valley_ekf, mid_t_ekf] = extract_peak_feature(tau_plot, resid_ekf, mid_window, 'min');
            [mid_valley_gpr, mid_t_gpr] = extract_peak_feature(tau_plot, resid_gpr, mid_window, 'min');

            pidx = find(selected_cycles == cycle_idx, 1, 'first');
            phase_label = selected_phase_labels{pidx};

            resid_ekf_i = interp1_monotonic_safe(tau_plot, resid_ekf, tauGrid);
            resid_gpr_i = interp1_monotonic_safe(tau_plot, resid_gpr, tauGrid);
            abs_err_ekf_i = interp1_monotonic_safe(tau_plot, abs_err_ekf, tauGrid);
            abs_err_gpr_i = interp1_monotonic_safe(tau_plot, abs_err_gpr, tauGrid);

            SelRows(end+1,1) = {table( ...
                battery_no, cycle_idx, string(phase_label), ...
                Q_nom_cycle_start / Q_nom_init_batt, ...
                qnom_was_clamped, ...
                rmse_ekf, rmse_gpr, ...
                rmse_gpr - rmse_ekf, ...
                res_energy_ekf, res_energy_gpr, ...
                late_peak_ekf, late_t_ekf, ...
                late_peak_gpr, late_t_gpr, ...
                mid_valley_ekf, mid_t_ekf, ...
                mid_valley_gpr, mid_t_gpr, ...
                rmse_V, ...
                'VariableNames', { ...
                'battery_no','cycle_idx','phase', ...
                'soh_proxy','qnom_was_clamped', ...
                'rmse_ekf','rmse_gpr', ...
                'delta_gpr_vs_ekf', ...
                'res_energy_ekf','res_energy_gpr', ...
                'late_peak_ekf','late_t_ekf', ...
                'late_peak_gpr','late_t_gpr', ...
                'mid_valley_ekf','mid_t_ekf', ...
                'mid_valley_gpr','mid_t_gpr', ...
                'rmse_v'})};

            TrajRows(end+1,1) = {table( ...
                repmat(battery_no, nGrid, 1), ...
                repmat(cycle_idx, nGrid, 1), ...
                repmat(string(phase_label), nGrid, 1), ...
                tauGrid, ...
                resid_ekf_i, resid_gpr_i, ...
                abs_err_ekf_i, abs_err_gpr_i, ...
                'VariableNames', { ...
                'battery_no','cycle_idx','phase','tau', ...
                'resid_ekf','resid_gpr', ...
                'abs_err_ekf','abs_err_gpr'})};

            Q_nom = Q_nom_next;

        catch ME
            fprintf('  Cycle %d error: %s\n', cycle_idx, ME.message);
            continue;
        end
    end
end

if isempty(SelRows) || isempty(TrajRows)
    error('No selected-cycle results were collected.');
end

GPRSelTable = vertcat(SelRows{:});
GPRTrajTable = vertcat(TrajRows{:});

GPRSelTable.phase = categorical(GPRSelTable.phase, phaseNames, 'Ordinal', true);
GPRTrajTable.phase = categorical(GPRTrajTable.phase, phaseNames, 'Ordinal', true);

if save_mat
    save(fullfile(outDir, 'GPRSelTable.mat'), 'GPRSelTable');
    save(fullfile(outDir, 'GPRTrajTable.mat'), 'GPRTrajTable');
end

%% =========================================================
% QUICK EXACT COUNT ON SELECTED CYCLES
%% =========================================================
eps0 = 1e-12;
num_gpr_worse = sum(GPRSelTable.delta_gpr_vs_ekf > eps0);
num_gpr_better = sum(GPRSelTable.delta_gpr_vs_ekf < -eps0);
num_gpr_equal = sum(abs(GPRSelTable.delta_gpr_vs_ekf) <= eps0);

fprintf('\n=========================================================\n');
fprintf('SELECTED-CYCLE COUNTS (FIRST/MIDDLE/LAST only)\n');
fprintf('Total selected cycles : %d\n', height(GPRSelTable));
fprintf('GPR better than EKF   : %d\n', num_gpr_better);
fprintf('GPR worse than EKF    : %d\n', num_gpr_worse);
fprintf('GPR equal to EKF      : %d\n', num_gpr_equal);
fprintf('=========================================================\n');

%% =========================================================
% FIGURE 1: SPAGHETTI RESIDUALS EKF vs GPR
%% =========================================================
f1 = figure('Position',[80 80 1550 430], 'Color','w');
tl = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

hGrey = gobjects(1);
hEKF  = gobjects(1);
hGPR  = gobjects(1);

for p = 1:3
    nexttile; hold on;

    Tp = GPRTrajTable(GPRTrajTable.phase == phaseNames{p}, :);
    ubc = unique(Tp(:, {'battery_no','cycle_idx'}));

    % individual EKF curves in grey
    for i = 1:height(ubc)
        idx = Tp.battery_no == ubc.battery_no(i) & Tp.cycle_idx == ubc.cycle_idx(i);

        htmp = plot(Tp.tau(idx), Tp.resid_ekf(idx), '-', ...
            'Color', cGrey, 'LineWidth', 1.1);

        % κρατάμε μόνο ένα handle για τη λεζάντα
        if p == 1 && i == 1
            hGrey = htmp;
        end
    end

    % means
    tau_u = unique(Tp.tau);
    m_ekf = aggregate_on_grid(Tp, 'resid_ekf', tau_u);
    m_gpr = aggregate_on_grid(Tp, 'resid_gpr', tau_u);

    h1 = plot(tau_u, m_ekf, '-', 'Color', cEKF, 'LineWidth', 2.3);
    h2 = plot(tau_u, m_gpr, '-', 'Color', cGPR, 'LineWidth', 2.8);

    if p == 1
        hEKF = h1;
        hGPR = h2;
    end

    yline(0,'k--','LineWidth',1.0);
    xlabel('Normalized cycle time');
    ylabel('SOC residual');
    title(sprintf('%s cycles', phaseNames{p}), 'FontWeight','bold');
    xlim([0 1]);
    grid on; box off;
end

lg = legend([hGrey, hEKF, hGPR], ...
    {'EKF individual', 'EKF mean', 'Raw GPR mean'}, ...
    'Location','southoutside', ...
    'Orientation','horizontal', ...
    'Box','off');
lg.Layout.Tile = 'south';

sgtitle('Spaghetti residuals: EKF baseline and raw GPR corrected mean', ...
    'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f1, outDir, 'Fig_G1_spaghetti_residuals_EKF_vs_GPR', save_png, save_pdf);

%% =========================================================
% FIGURE 2: LANDMARKS vs SOH
%
% open marker  = EKF
% filled marker = raw GPR
%% =========================================================
f2 = figure('Position',[80 80 1380 930], 'Color','w');
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% Late peak amplitude
nexttile; hold on;
plot_dual_scatter(GPRSelTable, 'soh_proxy', 'late_peak_ekf', 'late_peak_gpr', ...
    phaseNames, phaseColors, markers);
xlabel('SOH vs initial capacity');
ylabel('Late peak amplitude');
title('Late spike amplitude vs SOH', 'FontWeight','bold');
grid on; box off;

% Late peak timing
nexttile; hold on;
plot_dual_scatter(GPRSelTable, 'soh_proxy', 'late_t_ekf', 'late_t_gpr', ...
    phaseNames, phaseColors, markers);
xlabel('SOH vs initial capacity');
ylabel('Late peak timing (\tau)');
title('Late spike timing vs SOH', 'FontWeight','bold');
grid on; box off;

% Mid valley amplitude
nexttile; hold on;
plot_dual_scatter(GPRSelTable, 'soh_proxy', 'mid_valley_ekf', 'mid_valley_gpr', ...
    phaseNames, phaseColors, markers);
xlabel('SOH vs initial capacity');
ylabel('Mid-valley amplitude');
title('Mid-cycle valley amplitude vs SOH', 'FontWeight','bold');
grid on; box off;

% Mid valley timing
nexttile; hold on;
plot_dual_scatter(GPRSelTable, 'soh_proxy', 'mid_t_ekf', 'mid_t_gpr', ...
    phaseNames, phaseColors, markers);
xlabel('SOH vs initial capacity');
ylabel('Mid-valley timing (\tau)');
title('Mid-cycle valley timing vs SOH', 'FontWeight','bold');
grid on; box off;

lg = legend({'FIRST EKF','FIRST GPR','MIDDLE EKF','MIDDLE GPR','LAST EKF','LAST GPR'}, ...
    'Location','southoutside', 'Orientation','horizontal', 'Box','off');
lg.Layout.Tile = 'south';

sgtitle('Residual landmarks vs SOH: EKF (open) and raw GPR (filled)', ...
    'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f2, outDir, 'Fig_G2_landmarks_vs_SOH_EKF_vs_GPR', save_png, save_pdf);

%% =========================================================
% FIGURE 3: BATTERY-WISE TRANSITIONS EKF vs GPR
%% =========================================================
Btrans = build_transition_matrices(GPRSelTable, phaseNames);

f3 = figure('Position',[80 80 1300 900], 'Color','w');
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% RMSE
nexttile; hold on;
plot_transition_panel(Btrans.M_rmse_ekf, Btrans.M_rmse_gpr, phaseNames, cEKF, cGPR);
ylabel('RMSE');
title('RMSE EKF vs raw GPR', 'FontWeight','bold');
grid on; box off;

% Late peak amplitude
nexttile; hold on;
plot_transition_panel(Btrans.M_late_peak_ekf, Btrans.M_late_peak_gpr, phaseNames, cEKF, cGPR);
ylabel('Late peak amplitude');
title('Late peak amplitude transition', 'FontWeight','bold');
grid on; box off;

% Mid valley amplitude
nexttile; hold on;
plot_transition_panel(Btrans.M_mid_valley_ekf, Btrans.M_mid_valley_gpr, phaseNames, cEKF, cGPR);
ylabel('Mid-valley amplitude');
title('Mid-valley amplitude transition', 'FontWeight','bold');
grid on; box off;

% Residual energy
nexttile; hold on;
plot_transition_panel(Btrans.M_res_energy_ekf, Btrans.M_res_energy_gpr, phaseNames, cEKF, cGPR);
ylabel('Residual energy');
title('Residual energy transition', 'FontWeight','bold');
grid on; box off;

lg = legend({'Battery-wise EKF','Battery-wise GPR','EKF mean','GPR mean'}, ...
    'Location','southoutside', 'Orientation','horizontal', 'Box','off');
lg.Layout.Tile = 'south';

sgtitle('Battery-wise transitions across FIRST, MIDDLE, LAST cycles', ...
    'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f3, outDir, 'Fig_G3_batterywise_transitions_EKF_vs_GPR', save_png, save_pdf);

%% =========================================================
% FIGURE 4: ABSOLUTE ERROR CDFs
%% =========================================================
f4 = figure('Position',[80 80 1550 430], 'Color','w');
tl = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

for p = 1:3
    nexttile; hold on;

    Tp = GPRTrajTable(GPRTrajTable.phase == phaseNames{p}, :);

    plot_cdf(Tp.abs_err_ekf, cEKF);
    plot_cdf(Tp.abs_err_gpr, cGPR);

    xlabel('|SOC error|');
    ylabel('Empirical CDF');
    title(phaseNames{p}, 'FontWeight','bold');
    grid on; box off;
end

lg = legend({'EKF','Raw GPR'}, ...
    'Location','southoutside', 'Orientation','horizontal', 'Box','off');
lg.Layout.Tile = 'south';

sgtitle('Sample-level absolute SOC error distributions', ...
    'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f4, outDir, 'Fig_G4_abs_error_CDF_EKF_vs_GPR', save_png, save_pdf);

%% =========================================================
% CONSOLE SUMMARY
%% =========================================================
fprintf('\n=========================================================\n');
fprintf('EKF vs raw GPR sample-level plots generated.\n');
fprintf('Selected cycles analysed: %d\n', height(GPRSelTable));
fprintf('Batteries represented   : %d\n', numel(unique(GPRSelTable.battery_no)));
fprintf('Mean EKF RMSE           : %.6f\n', mean(GPRSelTable.rmse_ekf, 'omitnan'));
fprintf('Mean GPR RMSE           : %.6f\n', mean(GPRSelTable.rmse_gpr, 'omitnan'));
fprintf('Mean delta GPR-EKF      : %.6f\n', mean(GPRSelTable.delta_gpr_vs_ekf, 'omitnan'));
fprintf('Output folder           : %s\n', outDir);
fprintf('=========================================================\n');

%% =========================================================
% LOCAL FUNCTIONS
%% =========================================================
function tau = build_tau_plot(t)
    t = t(:);
    if isempty(t) || numel(t) < 2 || t(end) <= t(1)
        tau = linspace(0,1,numel(t))';
    else
        tau = (t - t(1)) / (t(end) - t(1));
    end
    tau(~isfinite(tau)) = 0;
    tau = min(max(tau,0),1);
end

function yq = interp1_monotonic_safe(x, y, xq)
    x = x(:); y = y(:); xq = xq(:);
    valid = isfinite(x) & isfinite(y);
    x = x(valid); y = y(valid);

    if numel(x) < 2
        yq = nan(size(xq));
        return;
    end

    [x, ia] = unique(x, 'stable');
    y = y(ia);

    if numel(x) < 2
        yq = nan(size(xq));
        return;
    end

    yq = interp1(x, y, xq, 'linear', 'extrap');
end

function [amp, tau0] = extract_peak_feature(tau, resid, window, modeStr)
    idx = tau >= window(1) & tau <= window(2) & isfinite(resid);
    if ~any(idx)
        amp = NaN;
        tau0 = NaN;
        return;
    end

    tau_w = tau(idx);
    res_w = resid(idx);

    switch lower(modeStr)
        case 'max'
            [amp, imax] = max(res_w);
            tau0 = tau_w(imax);
        case 'min'
            [amp, imin] = min(res_w);
            tau0 = tau_w(imin);
        otherwise
            error('Unknown modeStr.');
    end
end

function m = aggregate_on_grid(T, varName, tau_u)
    m = nan(size(tau_u));
    for i = 1:numel(tau_u)
        idx = T.tau == tau_u(i);
        vals = T.(varName)(idx);
        m(i) = mean(vals, 'omitnan');
    end
end

function plot_dual_scatter(T, xcol, ycol_ekf, ycol_gpr, phaseNames, phaseColors, markers)
    for p = 1:numel(phaseNames)
        idx = T.phase == phaseNames{p};

        scatter(T.(xcol)(idx), T.(ycol_ekf)(idx), 52, ...
            'Marker', markers{p}, ...
            'MarkerEdgeColor', phaseColors{p}, ...
            'MarkerFaceColor', 'w', ...
            'LineWidth', 1.2);

        scatter(T.(xcol)(idx), T.(ycol_gpr)(idx), 48, ...
            'Marker', markers{p}, ...
            'MarkerEdgeColor', phaseColors{p}, ...
            'MarkerFaceColor', phaseColors{p}, ...
            'MarkerFaceAlpha', 0.85, ...
            'LineWidth', 0.8);
    end
end

function B = build_transition_matrices(T, phaseNames)
    battList = unique(T.battery_no);
    nB = numel(battList);
    nP = numel(phaseNames);

    fnames = { ...
        'rmse_ekf','rmse_gpr', ...
        'late_peak_ekf','late_peak_gpr', ...
        'mid_valley_ekf','mid_valley_gpr', ...
        'res_energy_ekf','res_energy_gpr'};

    for f = 1:numel(fnames)
        B.(['M_' fnames{f}]) = nan(nB, nP);
    end

    B.battery_no = battList;

    for i = 1:nB
        Tb = T(T.battery_no == battList(i), :);
        for p = 1:nP
            idx = Tb.phase == phaseNames{p};
            if ~any(idx), continue; end
            row = find(idx,1,'first');
            for f = 1:numel(fnames)
                B.(['M_' fnames{f}])(i,p) = Tb.(fnames{f})(row);
            end
        end
    end
end

function plot_transition_panel(Mekf, Mgpr, phaseNames, cEKF, cGPR)
    x = 1:numel(phaseNames);

    for i = 1:size(Mekf,1)
        ye = Mekf(i,:);
        yg = Mgpr(i,:);

        plot(x, ye, '-o', 'Color', [0.80 0.80 0.80], 'LineWidth', 1.0, ...
            'MarkerSize', 4, 'MarkerFaceColor', 'w');
        plot(x, yg, '-o', 'Color', [0.82 0.88 1.00], 'LineWidth', 1.0, ...
            'MarkerSize', 4, 'MarkerFaceColor', 'w');
    end

    plot(x, mean(Mekf, 'omitnan'), '-o', 'Color', cEKF, ...
        'MarkerFaceColor', cEKF, 'MarkerSize', 8, 'LineWidth', 2.6);
    plot(x, mean(Mgpr, 'omitnan'), '-o', 'Color', cGPR, ...
        'MarkerFaceColor', 'w', 'MarkerSize', 9, 'LineWidth', 2.8);

    xlim([0.7 numel(phaseNames)+0.3]);
    xticks(x);
    xticklabels(lower(phaseNames));
end

% function plot_cdf(x, c)
%     x = x(isfinite(x));
%     if isempty(x), return; end
%     [f, xi] = ecdf(x);
%     plot(xi, f, 'Color', c, 'LineWidth', 2.2);
% end
% 
% function save_figure_multi(figHandle, outDir, baseName, save_png, save_pdf)
%     if save_png
%         exportgraphics(figHandle, fullfile(outDir, [baseName '.png']), 'Resolution', 300);
%     end
%     if save_pdf
%         exportgraphics(figHandle, fullfile(outDir, [baseName '.pdf']), 'ContentType', 'vector');
%     end
% end
% 
% function y = causalEMA(x, alpha)
%     x = x(:);
%     y = zeros(size(x));
%     if isempty(x), return; end
%     y(1) = x(1);
%     for k = 2:numel(x)
%         y(k) = alpha * x(k) + (1 - alpha) * y(k-1);
%     end
% end
% 
% function m = cummean_custom(x)
%     x = x(:);
%     m = zeros(size(x));
%     csum = 0;
%     cnt = 0;
%     for i = 1:numel(x)
%         if isfinite(x(i))
%             csum = csum + x(i);
%             cnt = cnt + 1;
%         end
%         if cnt > 0
%             m(i) = csum / cnt;
%         else
%             m(i) = NaN;
%         end
%     end
% end
% 
% function progress_causal = build_progress_causal(t, I_raw, Q_nom_cycle_start)
%     t = t(:);
%     I_raw = I_raw(:);
% 
%     N = numel(t);
%     progress_causal = zeros(N,1);
% 
%     if N < 2 || ~isfinite(Q_nom_cycle_start) || Q_nom_cycle_start <= 0
%         return;
%     end
% 
%     dt_sec = diff(t) * 60;
%     dt_sec(~isfinite(dt_sec) | dt_sec < 0) = 0;
% 
%     I_eff = 1.1 * I_raw(2:end);
%     dQ_abs = abs(I_eff .* dt_sec);
% 
%     Q_abs_cum = [0; cumsum(dQ_abs)];
% 
%     denom = max(2 * Q_nom_cycle_start, 1e-9);
%     progress_causal = Q_abs_cum / denom;
% 
%     progress_causal(~isfinite(progress_causal)) = 0;
%     progress_causal = min(max(progress_causal, 0), 1);
% end
% 
% function [Q_nom_next, was_clamped] = sanitize_qnom_next( ...
%     Q_nom_current, Q_accumulated, Q_nom_init_batt, ...
%     qnom_ratio_min, qnom_ratio_max, qnom_max_step_per_cycle)
% 
%     was_clamped = false;
%     Q_nom_next = Q_nom_current;
% 
%     if ~isfinite(Q_nom_current) || Q_nom_current <= 0 || ...
%        ~isfinite(Q_nom_init_batt) || Q_nom_init_batt <= 0
%         return;
%     end
% 
%     if ~isfinite(Q_accumulated) || Q_accumulated <= 0
%         return;
%     end
% 
%     qnom_candidate = Q_accumulated / 2;
%     if ~isfinite(qnom_candidate) || qnom_candidate <= 0
%         return;
%     end
% 
%     ratio_prev = Q_nom_current / Q_nom_init_batt;
%     ratio_cand = qnom_candidate / Q_nom_init_batt;
%     ratio_raw = ratio_cand;
% 
%     ratio_cand = min(max(ratio_cand, qnom_ratio_min), qnom_ratio_max);
% 
%     ratio_low  = max(qnom_ratio_min, ratio_prev - qnom_max_step_per_cycle);
%     ratio_high = min(qnom_ratio_max, ratio_prev + qnom_max_step_per_cycle);
% 
%     ratio_next = min(max(ratio_cand, ratio_low), ratio_high);
% 
%     if abs(ratio_next - ratio_raw) > 1e-12
%         was_clamped = true;
%     end
% 
%     Q_nom_next = ratio_next * Q_nom_init_batt;
% end
% 
% function OCV_func = OCV_func_local()
%     OCV_func = @(soc) interp1( ...
%         [0 0.02 0.05 0.10 0.20 0.40 0.60 0.80 0.90 0.95 0.98 1.00], ...
%         [2.00 2.75 3.05 3.18 3.24 3.27 3.29 3.31 3.33 3.35 3.39 3.60], ...
%         soc, 'pchip', 'extrap');
% end
% 
% function [SOC_est, V_model, rmse_V, Q_accumulated, debug] = ...
%     ekf_thevenin_2RC_R0_adaptive_v2(t, I, V_meas, Q_nom, R0, R1, C1, R2, C2, OCV_func)
% 
%     N = length(t);
%     Q_accumulated = 0;
% 
%     x = [0; 0; 0];
%     P = diag([1e-4, 1e-4, 1e-4]);
% 
%     q_soc  = 1e-7;
%     q_vrc1 = 1e-4;
%     q_vrc2 = 1e-4;
%     Qk = diag([q_soc, q_vrc1, q_vrc2]);
% 
%     Rk_base = 1e-5;
%     y_soc_freeze = 0.0223;
%     y_soc_full   = 0.035;
%     R_min = 1e-5;
%     R_max = 5e-2;
%     lambda_R = 0.999;
%     innov_var = Rk_base;
%     slope_thresh = 0.08;
%     slope_boost_factor = 6;
% 
%     SOC_est = zeros(N, 1);
%     V_model = 2 * ones(N, 1);
% 
%     debug.innovation = zeros(N,1);
%     debug.SOC_cc = zeros(N,1);
%     debug.dOCV_dSOC = zeros(N,1);
%     debug.Rk_eff = zeros(N,1);
%     debug.soc_gate_alpha = zeros(N,1);
% 
%     for k = 2:N
%         Ik = 1.1 * I(k);
%         delta_t = (t(k) - t(k-1)) * 60;
% 
%         Q_accumulated = Q_accumulated + abs(Ik * delta_t);
% 
%         SOC_pred = x(1) + (Ik * delta_t) / Q_nom;
%         SOC_pred = min(max(SOC_pred, 0), 1);
% 
%         a1 = exp(-delta_t / (R1 * C1));
%         a2 = exp(-delta_t / (R2 * C2));
% 
%         Vrc1_pred = a1 * x(2) + R1 * (1 - a1) * Ik;
%         Vrc2_pred = a2 * x(3) + R2 * (1 - a2) * Ik;
% 
%         x_pred = [SOC_pred; Vrc1_pred; Vrc2_pred];
% 
%         F = eye(3);
%         F(2,2) = a1;
%         F(3,3) = a2;
%         P_pred = F * P * F' + Qk;
% 
%         V_ocv_pred = OCV_func(SOC_pred);
%         V_pred = V_ocv_pred + R0 * Ik + Vrc1_pred + Vrc2_pred;
% 
%         dOCV_dSOC = numerical_dOCV_dSOC(SOC_pred, OCV_func);
%         H = [dOCV_dSOC, 1, 1];
% 
%         y = V_meas(k) - V_pred;
%         abs_y = abs(y);
% 
%         innov_var = lambda_R * innov_var + (1 - lambda_R) * (y^2);
%         Rk_eff = min(max(innov_var, R_min), R_max);
% 
%         if abs(dOCV_dSOC) < slope_thresh
%             Rk_eff = min(Rk_eff * slope_boost_factor, R_max);
%         end
% 
%         S = H * P_pred * H' + Rk_eff;
%         K = P_pred * H' / S;
% 
%         if abs_y <= y_soc_freeze
%             soc_gate_alpha = 0.0;
%         elseif abs_y >= y_soc_full
%             soc_gate_alpha = 1.0;
%         else
%             soc_gate_alpha = (abs_y - y_soc_freeze) / (y_soc_full - y_soc_freeze);
%         end
% 
%         dx = K * y;
%         dx(1) = soc_gate_alpha * dx(1);
% 
%         x = x_pred + dx;
%         x(1) = min(max(x(1), 0), 1);
% 
%         K_eff = K;
%         K_eff(1) = soc_gate_alpha * K_eff(1);
%         P = (eye(3) - K_eff * H) * P_pred;
% 
%         V_ocv_corr = OCV_func(x(1));
%         SOC_est(k) = x(1);
%         V_model(k) = min(max(V_ocv_corr + R0 * Ik + x(2) + x(3), 2.0), 3.6);
% 
%         debug.SOC_cc(k) = x_pred(1);
%         debug.innovation(k) = y;
%         debug.dOCV_dSOC(k) = dOCV_dSOC;
%         debug.Rk_eff(k) = Rk_eff;
%         debug.soc_gate_alpha(k) = soc_gate_alpha;
%     end
% 
%     rmse_V = sqrt(mean((V_meas - V_model).^2, 'omitnan'));
% end
% 
% function d = numerical_dOCV_dSOC(soc, OCV_func)
%     delta = 1e-5;
%     soc1 = max(0, min(1, soc - delta));
%     soc2 = max(0, min(1, soc + delta));
%     if abs(soc2 - soc1) < 1e-12
%         d = 0;
%     else
%         d = (OCV_func(soc2) - OCV_func(soc1)) / (soc2 - soc1);
%     end
% end
%%
%% =========================================================
% PLOT-ONLY SCRIPT: EKF vs RAW GPR
%
% Uses only:
%   - GPRSelTable.mat
%   - GPRTrajTable.mat
%
% No EKF rerun
% No GPR inference
% No raw battery data needed
%% =========================================================
%clc; close all;

%% =========================================================
% SETTINGS
%% =========================================================
outDir = 'gpr_vs_ekf_section32_figures';

save_png = true;
save_pdf = true;

set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultTextFontName', 'Arial');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultAxesLineWidth', 1.0);
set(groot, 'defaultLineLineWidth', 1.5);

% Colors
cEKF   = [0.12 0.12 0.12];
cGPR   = [0.00 0.15 1.00];
cGrey  = [0.72 0.72 0.72];

phaseNames = {'FIRST','MIDDLE','LAST'};
phaseColors = { ...
    [0.00 0.4470 0.7410], ...
    [0.8500 0.3250 0.0980], ...
    [0.9290 0.6940 0.1250]};
markers = {'o','s','d'};

%% =========================================================
% LOAD ONLY PRECOMPUTED TABLES
%% =========================================================
selFile  = fullfile(outDir, 'GPRSelTable.mat');
trajFile = fullfile(outDir, 'GPRTrajTable.mat');

if exist(selFile, 'file') ~= 2
    error('Missing file: %s', selFile);
end

if exist(trajFile, 'file') ~= 2
    error('Missing file: %s', trajFile);
end

S1 = load(selFile,  'GPRSelTable');
S2 = load(trajFile, 'GPRTrajTable');

if ~isfield(S1, 'GPRSelTable')
    error('GPRSelTable variable not found in %s', selFile);
end

if ~isfield(S2, 'GPRTrajTable')
    error('GPRTrajTable variable not found in %s', trajFile);
end

GPRSelTable  = S1.GPRSelTable;
GPRTrajTable = S2.GPRTrajTable;

% force categorical ordering if needed
GPRSelTable.phase = categorical(string(GPRSelTable.phase), phaseNames, 'Ordinal', true);
GPRTrajTable.phase = categorical(string(GPRTrajTable.phase), phaseNames, 'Ordinal', true);

fprintf('\n=========================================================\n');
fprintf('Loaded precomputed plot tables only.\n');
fprintf('Selected cycles : %d\n', height(GPRSelTable));
fprintf('Trajectory rows : %d\n', height(GPRTrajTable));
fprintf('Batteries       : %d\n', numel(unique(GPRSelTable.battery_no)));
fprintf('=========================================================\n');

%% =========================================================
% QUICK COUNTS
%% =========================================================
eps0 = 1e-12;
num_gpr_worse  = sum(GPRSelTable.delta_gpr_vs_ekf > eps0);
num_gpr_better = sum(GPRSelTable.delta_gpr_vs_ekf < -eps0);
num_gpr_equal  = sum(abs(GPRSelTable.delta_gpr_vs_ekf) <= eps0);

fprintf('\n=========================================================\n');
fprintf('SELECTED-CYCLE COUNTS (FIRST/MIDDLE/LAST only)\n');
fprintf('Total selected cycles : %d\n', height(GPRSelTable));
fprintf('GPR better than EKF   : %d\n', num_gpr_better);
fprintf('GPR worse than EKF    : %d\n', num_gpr_worse);
fprintf('GPR equal to EKF      : %d\n', num_gpr_equal);
fprintf('=========================================================\n');

%% =========================================================
% FIGURE 1: SPAGHETTI RESIDUALS EKF vs GPR
%% =========================================================
f1 = figure('Position',[80 80 1550 430], 'Color','w');
tl = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

hGrey = gobjects(1);
hEKF  = gobjects(1);
hGPR  = gobjects(1);

for p = 1:3
    nexttile; hold on;

    Tp = GPRTrajTable(GPRTrajTable.phase == phaseNames{p}, :);
    ubc = unique(Tp(:, {'battery_no','cycle_idx'}));

    for i = 1:height(ubc)
        idx = Tp.battery_no == ubc.battery_no(i) & Tp.cycle_idx == ubc.cycle_idx(i);

        htmp = plot(Tp.tau(idx), Tp.resid_ekf(idx), '-', ...
            'Color', cGrey, 'LineWidth', 1.1);

        if p == 1 && i == 1
            hGrey = htmp;
        end
    end

    tau_u = unique(Tp.tau);
    m_ekf = aggregate_on_grid(Tp, 'resid_ekf', tau_u);
    m_gpr = aggregate_on_grid(Tp, 'resid_gpr', tau_u);

    h1 = plot(tau_u, m_ekf, '-', 'Color', cEKF, 'LineWidth', 2.3);
    h2 = plot(tau_u, m_gpr, '-', 'Color', cGPR, 'LineWidth', 2.8);

    if p == 1
        hEKF = h1;
        hGPR = h2;
    end

    yline(0,'k--','LineWidth',1.0);
    xlabel('Normalized cycle time');
    ylabel('SOC residual');
    title(sprintf('%s cycles', phaseNames{p}), 'FontWeight','bold');
    xlim([0 1]);
    grid on; box off;
end

lg = legend([hGrey, hEKF, hGPR], ...
    {'EKF individual', 'EKF mean', 'Raw GPR mean'}, ...
    'Location','southoutside', ...
    'Orientation','horizontal', ...
    'Box','off');
lg.Layout.Tile = 'south';

%sgtitle('Spaghetti residuals: EKF baseline and raw GPR corrected mean', ...
    %'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f1, outDir, 'Fig_G1_spaghetti_residuals_EKF_vs_GPR', save_png, save_pdf);

%% =========================================================
% FIGURE 2: LANDMARKS vs SOH
%% =========================================================
f2 = figure('Position',[80 80 1380 930], 'Color','w');
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on;
plot_dual_scatter(GPRSelTable, 'soh_proxy', 'late_peak_ekf', 'late_peak_gpr', ...
    phaseNames, phaseColors, markers);
xlabel('SOH vs initial capacity');
ylabel('Late peak amplitude');
title('Late spike amplitude vs SOH', 'FontWeight','bold');
grid on; box off;

nexttile; hold on;
plot_dual_scatter(GPRSelTable, 'soh_proxy', 'late_t_ekf', 'late_t_gpr', ...
    phaseNames, phaseColors, markers);
xlabel('SOH vs initial capacity');
ylabel('Late peak timing (\tau)');
title('Late spike timing vs SOH', 'FontWeight','bold');
grid on; box off;

nexttile; hold on;
plot_dual_scatter(GPRSelTable, 'soh_proxy', 'mid_valley_ekf', 'mid_valley_gpr', ...
    phaseNames, phaseColors, markers);
xlabel('SOH vs initial capacity');
ylabel('Mid-valley amplitude');
title('Mid-cycle valley amplitude vs SOH', 'FontWeight','bold');
grid on; box off;

nexttile; hold on;
plot_dual_scatter(GPRSelTable, 'soh_proxy', 'mid_t_ekf', 'mid_t_gpr', ...
    phaseNames, phaseColors, markers);
xlabel('SOH vs initial capacity');
ylabel('Mid-valley timing (\tau)');
title('Mid-cycle valley timing vs SOH', 'FontWeight','bold');
grid on; box off;

lg = legend({'FIRST EKF','FIRST GPR','MIDDLE EKF','MIDDLE GPR','LAST EKF','LAST GPR'}, ...
    'Location','southoutside', 'Orientation','horizontal', 'Box','off');
lg.Layout.Tile = 'south';

% sgtitle('Residual landmarks vs SOH: EKF (open) and raw GPR (filled)', ...
%     'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f2, outDir, 'Fig_G2_landmarks_vs_SOH_EKF_vs_GPR', save_png, save_pdf);

%% =========================================================
% FIGURE 3: BATTERY-WISE TRANSITIONS EKF vs GPR
%% =========================================================
Btrans = build_transition_matrices(GPRSelTable, phaseNames);

f3 = figure('Position',[80 80 1300 900], 'Color','w');
tl = tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile; hold on;
plot_transition_panel(Btrans.M_rmse_ekf, Btrans.M_rmse_gpr, phaseNames, cEKF, cGPR);
ylabel('RMSE');
title('RMSE EKF vs raw GPR', 'FontWeight','bold');
grid on; box off;

nexttile; hold on;
plot_transition_panel(Btrans.M_late_peak_ekf, Btrans.M_late_peak_gpr, phaseNames, cEKF, cGPR);
ylabel('Late peak amplitude');
title('Late peak amplitude transition', 'FontWeight','bold');
grid on; box off;

nexttile; hold on;
plot_transition_panel(Btrans.M_mid_valley_ekf, Btrans.M_mid_valley_gpr, phaseNames, cEKF, cGPR);
ylabel('Mid-valley amplitude');
title('Mid-valley amplitude transition', 'FontWeight','bold');
grid on; box off;

nexttile; hold on;
plot_transition_panel(Btrans.M_res_energy_ekf, Btrans.M_res_energy_gpr, phaseNames, cEKF, cGPR);
ylabel('Residual energy');
title('Residual energy transition', 'FontWeight','bold');
grid on; box off;

lg = legend({'Battery-wise EKF','Battery-wise GPR'}, ...
    'Location','southoutside', 'Orientation','horizontal', 'Box','off');
lg.Layout.Tile = 'south';

% sgtitle('Battery-wise transitions across FIRST, MIDDLE, LAST cycles', ...
%     'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f3, outDir, 'Fig_G3_batterywise_transitions_EKF_vs_GPR', save_png, save_pdf);

%% =========================================================
% FIGURE 4: ABSOLUTE ERROR CDFs
%% =========================================================
f4 = figure('Position',[80 80 1550 430], 'Color','w');
tl = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

hCDF_EKF = gobjects(1);
hCDF_GPR = gobjects(1);

for p = 1:3
    nexttile; hold on;

    Tp = GPRTrajTable(GPRTrajTable.phase == phaseNames{p}, :);

    h1 = plot_cdf(Tp.abs_err_ekf, cEKF);
    h2 = plot_cdf(Tp.abs_err_gpr, cGPR);

    if p == 1
        hCDF_EKF = h1;
        hCDF_GPR = h2;
    end

    xlabel('|SOC error|');
    ylabel('Empirical CDF');
    title(phaseNames{p}, 'FontWeight','bold');
    grid on; box off;
end

lg = legend([hCDF_EKF, hCDF_GPR], {'EKF','Raw GPR'}, ...
    'Location','southoutside', 'Orientation','horizontal', 'Box','off');
lg.Layout.Tile = 'south';

% sgtitle('Sample-level absolute SOC error distributions', ...
%     'FontWeight','bold', 'FontSize', 14);

save_figure_multi(f4, outDir, 'Fig_G4_abs_error_CDF_EKF_vs_GPR', save_png, save_pdf);

%% =========================================================
% CONSOLE SUMMARY
%% =========================================================
fprintf('\n=========================================================\n');
fprintf('Plot-only run completed.\n');
fprintf('Selected cycles analysed: %d\n', height(GPRSelTable));
fprintf('Batteries represented   : %d\n', numel(unique(GPRSelTable.battery_no)));
fprintf('Mean EKF RMSE           : %.6f\n', mean(GPRSelTable.rmse_ekf, 'omitnan'));
fprintf('Mean GPR RMSE           : %.6f\n', mean(GPRSelTable.rmse_gpr, 'omitnan'));
fprintf('Mean delta GPR-EKF      : %.6f\n', mean(GPRSelTable.delta_gpr_vs_ekf, 'omitnan'));
fprintf('Output folder           : %s\n', outDir);
fprintf('=========================================================\n');

%% =========================================================
% LOCAL FUNCTIONS
%% =========================================================
% function m = aggregate_on_grid(T, varName, tau_u)
%     m = nan(size(tau_u));
%     for i = 1:numel(tau_u)
%         idx = T.tau == tau_u(i);
%         vals = T.(varName)(idx);
%         m(i) = mean(vals, 'omitnan');
%     end
% end
% 
% function plot_dual_scatter(T, xcol, ycol_ekf, ycol_gpr, phaseNames, phaseColors, markers)
%     for p = 1:numel(phaseNames)
%         idx = T.phase == phaseNames{p};
% 
%         scatter(T.(xcol)(idx), T.(ycol_ekf)(idx), 52, ...
%             'Marker', markers{p}, ...
%             'MarkerEdgeColor', phaseColors{p}, ...
%             'MarkerFaceColor', 'w', ...
%             'LineWidth', 1.2);
% 
%         scatter(T.(xcol)(idx), T.(ycol_gpr)(idx), 48, ...
%             'Marker', markers{p}, ...
%             'MarkerEdgeColor', phaseColors{p}, ...
%             'MarkerFaceColor', phaseColors{p}, ...
%             'MarkerFaceAlpha', 0.85, ...
%             'LineWidth', 0.8);
%     end
% end
% 
% function B = build_transition_matrices(T, phaseNames)
%     battList = unique(T.battery_no);
%     nB = numel(battList);
%     nP = numel(phaseNames);
% 
%     fnames = { ...
%         'rmse_ekf','rmse_gpr', ...
%         'late_peak_ekf','late_peak_gpr', ...
%         'mid_valley_ekf','mid_valley_gpr', ...
%         'res_energy_ekf','res_energy_gpr'};
% 
%     for f = 1:numel(fnames)
%         B.(['M_' fnames{f}]) = nan(nB, nP);
%     end
% 
%     B.battery_no = battList;
% 
%     for i = 1:nB
%         Tb = T(T.battery_no == battList(i), :);
%         for p = 1:nP
%             idx = Tb.phase == phaseNames{p};
%             if ~any(idx), continue; end
%             row = find(idx,1,'first');
%             for f = 1:numel(fnames)
%                 B.(['M_' fnames{f}])(i,p) = Tb.(fnames{f})(row);
%             end
%         end
%     end
% end
% 
% function plot_transition_panel(Mekf, Mgpr, phaseNames, cEKF, cGPR)
%     x = 1:numel(phaseNames);
% 
%     for i = 1:size(Mekf,1)
%         ye = Mekf(i,:);
%         yg = Mgpr(i,:);
% 
%         plot(x, ye, '-o', 'Color', [0.80 0.80 0.80], 'LineWidth', 1.0, ...
%             'MarkerSize', 4, 'MarkerFaceColor', 'w');
%         plot(x, yg, '-o', 'Color', [0.82 0.88 1.00], 'LineWidth', 1.0, ...
%             'MarkerSize', 4, 'MarkerFaceColor', 'w');
%     end
% 
%     plot(x, mean(Mekf, 'omitnan'), '-o', 'Color', cEKF, ...
%         'MarkerFaceColor', cEKF, 'MarkerSize', 8, 'LineWidth', 2.6);
%     plot(x, mean(Mgpr, 'omitnan'), '-o', 'Color', cGPR, ...
%         'MarkerFaceColor', 'w', 'MarkerSize', 9, 'LineWidth', 2.8);
% 
%     xlim([0.7 numel(phaseNames)+0.3]);
%     xticks(x);
%     xticklabels(lower(phaseNames));
% end
% 
% function h = plot_cdf(x, c)
%     x = x(isfinite(x));
%     if isempty(x)
%         h = gobjects(1);
%         return;
%     end
%     [f, xi] = ecdf(x);
%     h = plot(xi, f, 'Color', c, 'LineWidth', 2.2);
% end
% 
% function save_figure_multi(figHandle, outDir, baseName, save_png, save_pdf)
%     if ~exist(outDir, 'dir')
%         mkdir(outDir);
%     end
%     if save_png
%         exportgraphics(figHandle, fullfile(outDir, [baseName '.png']), 'Resolution', 300);
%     end
%     if save_pdf
%         exportgraphics(figHandle, fullfile(outDir, [baseName '.pdf']), 'ContentType', 'vector');
%     end
% end