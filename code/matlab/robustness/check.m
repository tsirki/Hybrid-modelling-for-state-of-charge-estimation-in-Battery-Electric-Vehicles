%% =========================================================
% VARIANT C-LITE ROBUSTNESS TESTING
% EKF vs RAW GPR ONLY
%
% SCENARIOS
%   1) baseline
%   2) init_soc_plus5   : +5% initial SOC error
%   3) high_noise       : stronger current / voltage noise
%   4) current_bias     : constant current sensor bias
%
% GOAL
%   Evaluate robustness of the already-trained Variant C-Lite raw GPR
%   correction against the EKF under stressed conditions, while:
%     - using PRIMARY test batteries only
%     - keeping the same q_nom clamp logic
%     - saving cycle-level, battery-level, selected-cycle and
%       trajectory-level tables for later comparative plots
%
% IMPORTANT
%   - No retraining
%   - No Lite model
%   - No shape clustering here
%
% REQUIRED FILES (if vars not already in workspace)
%   - battery_workspace_core.mat
%   - fusion_full_model.mat
%   - Q_nom_init_first_cycle_all_batteries.mat
%   - gpr_variantC_lite_model.mat
%
% OUTPUTS
%   - robustness_cycle_results_all.csv
%   - robustness_battery_summary_all.csv
%   - robustness_scenario_summary.csv
%   - robustness_selected_cycles.csv
%   - robustness_outputs.mat
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
    if exist('battery_workspace_core.mat','file') == 2
        load('battery_workspace_core.mat', 't_all','I_all','V_all','Q_all');
    else
        error('Missing core vars and battery_workspace_core.mat not found.');
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
        load('fusion_full_model.mat', 'I_noise_std','V_noise_std','R0','R1','C1','R2','C2');
    else
        error('Missing fusion vars and fusion_full_model.mat not found.');
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
min_cycle_length = 30;
residual_clamp   = 0.03;

% q_nom clamp
qnom_ratio_min = 0.60;
qnom_ratio_max = 1.05;
qnom_max_step_per_cycle = 0.03;

% selected-cycle exports for later plots
save_selected_cycle_tables = true;
nGrid = 201;
tauGrid_plot = linspace(0,1,nGrid)';

late_window = [0.70 0.98];
mid_window  = [0.15 0.60];
phaseNames = {'FIRST','MIDDLE','LAST'};

% filenames
save_csv = true;
save_mat = true;

cycle_csv_name    = 'robustness_cycle_results_all.csv';
battery_csv_name  = 'robustness_battery_summary_all.csv';
scenario_csv_name = 'robustness_scenario_summary.csv';
selected_csv_name = 'robustness_selected_cycles.csv';
outputs_mat_name  = 'robustness_outputs.mat';

%% =========================================================
% PRIMARY TEST SPLIT
%% =========================================================
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

primary_pool = 1:min(84, numel(t_all));
exclude_batteries = unique([train_batteries_eval(:); 15]);
primary_test_batteries = setdiff(primary_pool, exclude_batteries, 'stable');
test_batteries = primary_test_batteries(:)';

fprintf('\n=========================================================\n');
fprintf('ROBUSTNESS TESTING: EKF vs RAW GPR\n');
fprintf('Primary test batteries:\n');
fprintf('%s\n', mat2str(primary_test_batteries));
fprintf('Count = %d\n', numel(primary_test_batteries));
fprintf('=========================================================\n');

%% =========================================================
% SCENARIO DEFINITIONS
%% =========================================================
ScenarioList = struct([]);

% baseline
ScenarioList(1).name = "baseline";
ScenarioList(1).init_soc_offset = 0.00;
ScenarioList(1).i_noise_mult = 1.0;
ScenarioList(1).v_noise_mult = 1.0;
ScenarioList(1).current_bias_frac = 0.00;

% +5% initial SOC error
ScenarioList(2).name = "init_soc_plus5";
ScenarioList(2).init_soc_offset = +0.05;
ScenarioList(2).i_noise_mult = 1.0;
ScenarioList(2).v_noise_mult = 1.0;
ScenarioList(2).current_bias_frac = 0.00;

% stronger measurement noise
ScenarioList(3).name = "high_noise";
ScenarioList(3).init_soc_offset = 0.00;
ScenarioList(3).i_noise_mult = 1.5;
ScenarioList(3).v_noise_mult = 1.5;
ScenarioList(3).current_bias_frac = 0.00;

% current sensor bias
ScenarioList(4).name = "current_bias";
ScenarioList(4).init_soc_offset = 0.00;
ScenarioList(4).i_noise_mult = 1.0;
ScenarioList(4).v_noise_mult = 1.0;
ScenarioList(4).current_bias_frac = 0.005;   % 2% multiplicative bias

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

%% =========================================================
% MAIN LOOP
%% =========================================================
CycleResultsAll   = table();
BatterySummaryAll = table();
ScenarioSummary   = table();

RobustSelRows  = {};
RobustTrajRows = {};

for iscn = 1:numel(ScenarioList)

    SC = ScenarioList(iscn);
    fprintf('\n=========================================================\n');
    fprintf('Scenario %d / %d : %s\n', iscn, numel(ScenarioList), SC.name);
    fprintf('init_soc_offset   = %+0.3f\n', SC.init_soc_offset);
    fprintf('i_noise_mult      = %.2f\n', SC.i_noise_mult);
    fprintf('v_noise_mult      = %.2f\n', SC.v_noise_mult);
    fprintf('current_bias_frac = %+0.3f\n', SC.current_bias_frac);
    fprintf('=========================================================\n');

    CycleResultsScenario = table();
    BatterySummaryScenario = table();

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
        selected_cycles = selected_cycles(selected_cycles >= 1 & selected_cycles <= num_cycles);

        fprintf('\nBattery %d | num_cycles = %d\n', battery_no, num_cycles);

        RowCell = cell(num_cycles,1);
        row_count = 0;

        rmse_ekf_all = [];
        rmse_gpr_all = [];
        rmse_ekf_late_all = [];
        rmse_gpr_late_all = [];
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

                % stressed measurements
                I_meas = I_data .* (1 + SC.current_bias_frac) + ...
                    (SC.i_noise_mult * I_noise_std) * randn(size(I_data));

                V_meas = V_data + ...
                    (SC.v_noise_mult * V_noise_std) * randn(size(V_data));

                Q_nom_cycle_start = Q_nom;

                [SOC_est, V_model, rmse_V, Q_accumulated, debug] = ...
                    ekf_thevenin_2RC_R0_adaptive_v2_stress( ...
                        t_data, I_meas, V_meas, ...
                        Q_nom, R0, R1, C1, R2, C2, ...
                        OCV_func_local(), SC.init_soc_offset);

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

                % raw GPR correction
                SOC_gpr = SOC_est + residual_pred;
                SOC_gpr = min(max(SOC_gpr, 0), 1);

                % metrics
                rmse_ekf = sqrt(mean((SOC_est - SOC_true).^2, 'omitnan'));
                rmse_gpr = sqrt(mean((SOC_gpr - SOC_true).^2, 'omitnan'));

                lateMask = progress_causal >= 0.80;
                if any(lateMask)
                    rmse_ekf_late = sqrt(mean((SOC_est(lateMask) - SOC_true(lateMask)).^2, 'omitnan'));
                    rmse_gpr_late = sqrt(mean((SOC_gpr(lateMask) - SOC_true(lateMask)).^2, 'omitnan'));
                else
                    rmse_ekf_late = NaN;
                    rmse_gpr_late = NaN;
                end

                delta_gpr_vs_ekf = rmse_gpr - rmse_ekf;
                delta_gpr_vs_ekf_late = rmse_gpr_late - rmse_ekf_late;
                is_gpr_improved_vs_ekf = double(rmse_gpr < rmse_ekf);
                is_gpr_worse_vs_ekf    = double(rmse_gpr > rmse_ekf);

                rmse_ekf_all = [rmse_ekf_all; rmse_ekf]; %#ok<AGROW>
                rmse_gpr_all = [rmse_gpr_all; rmse_gpr]; %#ok<AGROW>
                rmse_ekf_late_all = [rmse_ekf_late_all; rmse_ekf_late]; %#ok<AGROW>
                rmse_gpr_late_all = [rmse_gpr_late_all; rmse_gpr_late]; %#ok<AGROW>
                qnom_clamped_all = [qnom_clamped_all; qnom_was_clamped]; %#ok<AGROW>

                is_selected_cycle = double(ismember(cycle_idx, selected_cycles));

                row_count = row_count + 1;
                RowCell{row_count} = table( ...
                    string(SC.name), battery_no, cycle_idx, ...
                    Q_nom_cycle_start / Q_nom_init_batt, ...
                    qnom_was_clamped, ...
                    SC.init_soc_offset, SC.i_noise_mult, SC.v_noise_mult, SC.current_bias_frac, ...
                    rmse_ekf, rmse_gpr, ...
                    delta_gpr_vs_ekf, ...
                    rmse_ekf_late, rmse_gpr_late, ...
                    delta_gpr_vs_ekf_late, ...
                    mean(abs(residual_pred), 'omitnan'), ...
                    rmse_V, ...
                    is_gpr_improved_vs_ekf, ...
                    is_gpr_worse_vs_ekf, ...
                    is_selected_cycle, ...
                    'VariableNames', { ...
                    'scenario_name','battery_no','cycle_idx','qnom_start_frac', ...
                    'qnom_was_clamped', ...
                    'init_soc_offset','i_noise_mult','v_noise_mult','current_bias_frac', ...
                    'rmse_ekf','rmse_gpr', ...
                    'delta_gpr_vs_ekf', ...
                    'rmse_ekf_late','rmse_gpr_late', ...
                    'delta_gpr_vs_ekf_late', ...
                    'mean_abs_residual_pred','rmse_v', ...
                    'is_gpr_improved_vs_ekf', ...
                    'is_gpr_worse_vs_ekf', ...
                    'is_selected_cycle'});

                % selected-cycle exports for later plotting
                if save_selected_cycle_tables && is_selected_cycle
                    abs_err_ekf = abs(SOC_true - SOC_est);
                    abs_err_gpr = abs(SOC_true - SOC_gpr);

                    tau_cycle = build_tau_plot(t);
                    res_energy_ekf = trapz(tau_cycle, (SOC_true - SOC_est).^2);
                    res_energy_gpr = trapz(tau_cycle, (SOC_true - SOC_gpr).^2);

                    [late_peak_ekf, late_t_ekf] = extract_peak_feature(tau_cycle, SOC_true - SOC_est, late_window, 'max');
                    [late_peak_gpr, late_t_gpr] = extract_peak_feature(tau_cycle, SOC_true - SOC_gpr, late_window, 'max');

                    [mid_valley_ekf, mid_t_ekf] = extract_peak_feature(tau_cycle, SOC_true - SOC_est, mid_window, 'min');
                    [mid_valley_gpr, mid_t_gpr] = extract_peak_feature(tau_cycle, SOC_true - SOC_gpr, mid_window, 'min');

                    pidx = find(selected_cycles == cycle_idx, 1, 'first');
                    if numel(selected_cycles) == 1
                        phase_label = 'MIDDLE';
                    elseif numel(selected_cycles) == 2
                        phase_label = ternary(pidx == 1, 'FIRST', 'LAST');
                    else
                        phase_label = phaseNames{pidx};
                    end

                    resid_ekf_i   = interp1_monotonic_safe(tau_cycle, SOC_true - SOC_est, tauGrid_plot);
                    resid_gpr_i   = interp1_monotonic_safe(tau_cycle, SOC_true - SOC_gpr, tauGrid_plot);
                    abs_err_ekf_i = interp1_monotonic_safe(tau_cycle, abs_err_ekf, tauGrid_plot);
                    abs_err_gpr_i = interp1_monotonic_safe(tau_cycle, abs_err_gpr, tauGrid_plot);

                    soc_true_i = interp1_monotonic_safe(tau_cycle, SOC_true, tauGrid_plot);
                    soc_ekf_i  = interp1_monotonic_safe(tau_cycle, SOC_est,  tauGrid_plot);
                    soc_gpr_i  = interp1_monotonic_safe(tau_cycle, SOC_gpr,  tauGrid_plot);

                    v_true_i  = interp1_monotonic_safe(tau_cycle, V_raw,   tauGrid_plot);
                    v_model_i = interp1_monotonic_safe(tau_cycle, V_model, tauGrid_plot);

                    RobustSelRows(end+1,1) = {table( ...
                        string(SC.name), battery_no, cycle_idx, string(phase_label), ...
                        Q_nom_cycle_start / Q_nom_init_batt, ...
                        qnom_was_clamped, ...
                        SC.init_soc_offset, SC.i_noise_mult, SC.v_noise_mult, SC.current_bias_frac, ...
                        rmse_ekf, rmse_gpr, ...
                        delta_gpr_vs_ekf, ...
                        res_energy_ekf, res_energy_gpr, ...
                        late_peak_ekf, late_t_ekf, ...
                        late_peak_gpr, late_t_gpr, ...
                        mid_valley_ekf, mid_t_ekf, ...
                        mid_valley_gpr, mid_t_gpr, ...
                        rmse_V, ...
                        'VariableNames', { ...
                        'scenario_name','battery_no','cycle_idx','phase', ...
                        'soh_proxy','qnom_was_clamped', ...
                        'init_soc_offset','i_noise_mult','v_noise_mult','current_bias_frac', ...
                        'rmse_ekf','rmse_gpr', ...
                        'delta_gpr_vs_ekf', ...
                        'res_energy_ekf','res_energy_gpr', ...
                        'late_peak_ekf','late_t_ekf', ...
                        'late_peak_gpr','late_t_gpr', ...
                        'mid_valley_ekf','mid_t_ekf', ...
                        'mid_valley_gpr','mid_t_gpr', ...
                        'rmse_v'})};

                    RobustTrajRows(end+1,1) = {table( ...
                        repmat(string(SC.name), nGrid, 1), ...
                        repmat(battery_no, nGrid, 1), ...
                        repmat(cycle_idx, nGrid, 1), ...
                        repmat(string(phase_label), nGrid, 1), ...
                        tauGrid_plot, ...
                        soc_true_i, ...
                        soc_ekf_i, ...
                        soc_gpr_i, ...
                        v_true_i, ...
                        v_model_i, ...
                        resid_ekf_i, ...
                        resid_gpr_i, ...
                        abs_err_ekf_i, ...
                        abs_err_gpr_i, ...
                        'VariableNames', { ...
                        'scenario_name','battery_no','cycle_idx','phase','tau', ...
                        'soc_true','soc_ekf','soc_gpr', ...
                        'v_true','v_model', ...
                        'resid_ekf','resid_gpr', ...
                        'abs_err_ekf','abs_err_gpr'})};
                end

                Q_nom = Q_nom_next;

            catch err
                fprintf('Battery %d | cycle %d error: %s\n', battery_no, cycle_idx, err.message);
                continue;
            end
        end

        if row_count > 0
            CycleResultsScenario = [CycleResultsScenario; vertcat(RowCell{1:row_count})]; %#ok<AGROW>
        end

        if ~isempty(rmse_ekf_all)
            brow = table( ...
                string(SC.name), battery_no, ...
                numel(rmse_ekf_all), ...
                mean(rmse_ekf_all,'omitnan'), ...
                mean(rmse_gpr_all,'omitnan'), ...
                mean(rmse_ekf_late_all,'omitnan'), ...
                mean(rmse_gpr_late_all,'omitnan'), ...
                sum(rmse_gpr_all < rmse_ekf_all,'omitnan'), ...
                sum(rmse_gpr_all > rmse_ekf_all,'omitnan'), ...
                mean(rmse_gpr_all < rmse_ekf_all,'omitnan'), ...
                mean(rmse_gpr_all > rmse_ekf_all,'omitnan'), ...
                sum(qnom_clamped_all,'omitnan'), ...
                'VariableNames', { ...
                'scenario_name','battery_no','num_cycles_evaluated', ...
                'mean_rmse_ekf','mean_rmse_gpr', ...
                'mean_rmse_ekf_late','mean_rmse_gpr_late', ...
                'num_gpr_improved_vs_ekf', ...
                'num_gpr_worse_vs_ekf', ...
                'gpr_improve_rate_vs_ekf', ...
                'gpr_failure_rate_vs_ekf', ...
                'num_qnom_clamped_cycles'});
            BatterySummaryScenario = [BatterySummaryScenario; brow]; %#ok<AGROW>
        end
    end

    if ~isempty(CycleResultsScenario)
        CycleResultsScenario = sortrows(CycleResultsScenario, {'battery_no','cycle_idx'});
    end

    if ~isempty(BatterySummaryScenario)
        BatterySummaryScenario = sortrows(BatterySummaryScenario, 'battery_no');
    end

    CycleResultsAll   = [CycleResultsAll; CycleResultsScenario]; %#ok<AGROW>
    BatterySummaryAll = [BatterySummaryAll; BatterySummaryScenario]; %#ok<AGROW>

    if ~isempty(BatterySummaryScenario)
        srow = table( ...
            string(SC.name), ...
            SC.init_soc_offset, SC.i_noise_mult, SC.v_noise_mult, SC.current_bias_frac, ...
            height(BatterySummaryScenario), ...
            sum(BatterySummaryScenario.num_cycles_evaluated,'omitnan'), ...
            mean(BatterySummaryScenario.mean_rmse_ekf,'omitnan'), ...
            mean(BatterySummaryScenario.mean_rmse_gpr,'omitnan'), ...
            mean(BatterySummaryScenario.mean_rmse_gpr,'omitnan') - mean(BatterySummaryScenario.mean_rmse_ekf,'omitnan'), ...
            mean(BatterySummaryScenario.mean_rmse_ekf_late,'omitnan'), ...
            mean(BatterySummaryScenario.mean_rmse_gpr_late,'omitnan'), ...
            mean(BatterySummaryScenario.gpr_improve_rate_vs_ekf,'omitnan'), ...
            mean(BatterySummaryScenario.gpr_failure_rate_vs_ekf,'omitnan'), ...
            'VariableNames', { ...
            'scenario_name', ...
            'init_soc_offset','i_noise_mult','v_noise_mult','current_bias_frac', ...
            'num_batteries','num_cycles', ...
            'mean_rmse_ekf','mean_rmse_gpr','mean_delta_gpr_vs_ekf', ...
            'mean_rmse_ekf_late','mean_rmse_gpr_late', ...
            'mean_gpr_improve_rate_vs_ekf','mean_gpr_failure_rate_vs_ekf'});
        ScenarioSummary = [ScenarioSummary; srow]; %#ok<AGROW>
    end
end

%% =========================================================
% FINAL ASSEMBLY
%% =========================================================
if ~isempty(CycleResultsAll)
    CycleResultsAll = sortrows(CycleResultsAll, {'scenario_name','battery_no','cycle_idx'});
end

if ~isempty(BatterySummaryAll)
    BatterySummaryAll = sortrows(BatterySummaryAll, {'scenario_name','battery_no'});
end

if ~isempty(ScenarioSummary)
    ScenarioSummary = sortrows(ScenarioSummary, 'scenario_name');
end

if ~isempty(RobustSelRows)
    RobustSelTable = vertcat(RobustSelRows{:});
    RobustSelTable.phase = categorical(string(RobustSelTable.phase), phaseNames, 'Ordinal', true);
else
    RobustSelTable = table();
end

if ~isempty(RobustTrajRows)
    RobustTrajTable = vertcat(RobustTrajRows{:});
    RobustTrajTable.phase = categorical(string(RobustTrajTable.phase), phaseNames, 'Ordinal', true);
else
    RobustTrajTable = table();
end

%% =========================================================
% DISPLAY
%% =========================================================
disp(' ');
disp('==================== SCENARIO SUMMARY ====================');
if ~isempty(ScenarioSummary)
    disp(ScenarioSummary);
else
    disp('ScenarioSummary is empty.');
end

disp(' ');
disp('==================== CYCLE RESULTS HEAD ====================');
if ~isempty(CycleResultsAll)
    disp(CycleResultsAll(1:min(20,height(CycleResultsAll)), :));
else
    disp('CycleResultsAll is empty.');
end

fprintf('\n==================== ROBUSTNESS SUMMARY ====================\n');
for i = 1:height(ScenarioSummary)
    fprintf(['%s | mean EKF = %.6f | mean GPR = %.6f | mean Δ(GPR-EKF) = %.6f | ' ...
             'improve rate = %.2f %% | failure rate = %.2f %%\n'], ...
        ScenarioSummary.scenario_name(i), ...
        ScenarioSummary.mean_rmse_ekf(i), ...
        ScenarioSummary.mean_rmse_gpr(i), ...
        ScenarioSummary.mean_delta_gpr_vs_ekf(i), ...
        100*ScenarioSummary.mean_gpr_improve_rate_vs_ekf(i), ...
        100*ScenarioSummary.mean_gpr_failure_rate_vs_ekf(i));
end

%% =========================================================
% SAVE CSV
%% =========================================================
if save_csv
    if ~isempty(CycleResultsAll)
        writetable(CycleResultsAll, cycle_csv_name);
    else
        writetable(table(), cycle_csv_name);
    end

    if ~isempty(BatterySummaryAll)
        writetable(BatterySummaryAll, battery_csv_name);
    else
        writetable(table(), battery_csv_name);
    end

    if ~isempty(ScenarioSummary)
        writetable(ScenarioSummary, scenario_csv_name);
    else
        writetable(table(), scenario_csv_name);
    end

    if ~isempty(RobustSelTable)
        writetable(RobustSelTable, selected_csv_name);
    else
        writetable(table(), selected_csv_name);
    end

    fprintf('\nSaved CSV files:\n');
    fprintf('  %s\n', cycle_csv_name);
    fprintf('  %s\n', battery_csv_name);
    fprintf('  %s\n', scenario_csv_name);
    fprintf('  %s\n', selected_csv_name);
end

%% =========================================================
% SAVE MAT
%% =========================================================
if save_mat
    save(outputs_mat_name, ...
        'CycleResultsAll', ...
        'BatterySummaryAll', ...
        'ScenarioSummary', ...
        'RobustSelTable', ...
        'RobustTrajTable', ...
        'ScenarioList', ...
        'test_batteries', ...
        'primary_test_batteries', ...
        'train_batteries_eval', ...
        'exclude_batteries', ...
        'predictors', ...
        'qnom_ratio_min', ...
        'qnom_ratio_max', ...
        'qnom_max_step_per_cycle', ...
        'residual_clamp', ...
        'min_cycle_length', ...
        '-v7.3');

    fprintf('\nSaved MAT output:\n');
    fprintf('  %s\n', outputs_mat_name);
end

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
    ratio_raw  = ratio_cand;

    ratio_cand = min(max(ratio_cand, qnom_ratio_min), qnom_ratio_max);

    ratio_low  = max(qnom_ratio_min, ratio_prev - qnom_max_step_per_cycle);
    ratio_high = min(qnom_ratio_max, ratio_prev + qnom_max_step_per_cycle);

    ratio_next = min(max(ratio_cand, ratio_low), ratio_high);

    if abs(ratio_next - ratio_raw) > 1e-12
        was_clamped = true;
    end

    Q_nom_next = ratio_next * Q_nom_init_batt;
end

function OCV_func = OCV_func_local()
    OCV_func = @(soc) interp1( ...
        [0 0.02 0.05 0.10 0.20 0.40 0.60 0.80 0.90 0.95 0.98 1.00], ...
        [2.00 2.75 3.05 3.18 3.24 3.27 3.29 3.31 3.33 3.35 3.39 3.60], ...
        soc, 'pchip', 'extrap');
end

function [SOC_est, V_model, rmse_V, Q_accumulated, debug] = ...
    ekf_thevenin_2RC_R0_adaptive_v2_stress( ...
    t, I, V_meas, Q_nom, R0, R1, C1, R2, C2, OCV_func, init_soc_offset)

    N = length(t);
    Q_accumulated = 0;

    x0_soc = min(max(init_soc_offset, 0), 1);
    x = [x0_soc; 0; 0];
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

    SOC_est(1) = x(1);

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

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end