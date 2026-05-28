%% =========================================================
% ADAPTIVE BAYESIAN FUSION + PLOT PACKAGE EXPORT
%
% GOAL:
%   Evaluate adaptive Bayesian fusion and save all the required
%   tables so we can later generate the same kind of plots as before:
%     - EKF vs Bayes vs True SOC
%     - spaghetti residual plots
%     - CDF absolute error
%     - transition plots
%     - landmark plots
%
% MAIN OUTPUTS:
%   adaptive_bayes_cycle_results.csv
%   adaptive_bayes_battery_summary.csv
%   adaptive_bayes_outputs.mat
%
% PLOT PACKAGE OUTPUTS:
%   adaptive_bayes_plot_package.mat
%   BayesSelTable
%   BayesTrajTable
%
% IMPORTANT:
%   - no retraining
%   - no saving of I_all / V_all / Q_all / t_all
%   - saves only compact plotting tables
%% =========================================================

%clearvars;
clc; close all;
rng(42);

%% =========================================================
% USER SETTINGS
%% =========================================================

% ---------------------------------------------------------
% BATTERY SELECTION
% ---------------------------------------------------------
use_secondary_default = true;     
custom_test_batteries = [101 105 124];

% ---------------------------------------------------------
% PLOT REQUESTS
% each row = [battery_no cycle_idx]
% ---------------------------------------------------------
make_requested_plots = true;
plot_mode = 'full';   % 'full' or 'residual_only'
plot_requests = [
    101    2
    101  600
    124 1200
];

save_plot_files = true;
plot_dir = 'adaptive_bayes_plots';

% ---------------------------------------------------------
% SAVE OPTIONS
% ---------------------------------------------------------
save_csv = true;
save_mat = true;
save_partial_mat = true;
save_plot_package = true;

save_per_battery_csv = true;
save_per_battery_mat = false;
per_battery_dir = 'adaptive_bayes_per_battery';

cycle_csv_name         = 'adaptive_bayes_cycle_results.csv';
summary_csv_name       = 'adaptive_bayes_battery_summary.csv';
outputs_mat_name       = 'adaptive_bayes_outputs.mat';
partial_outputs_mat    = 'adaptive_bayes_outputs_partial.mat';
plot_package_mat_name  = 'adaptive_bayes_plot_package.mat';

% ---------------------------------------------------------
% RUNTIME / LOGGING
% ---------------------------------------------------------
print_every_n_cycles = 50;

% ---------------------------------------------------------
% DATA / MODEL SETTINGS
% ---------------------------------------------------------
min_cycle_length = 30;
residual_clamp = 0.03;

% Q_nom clamp
qnom_ratio_min = 0.60;
qnom_ratio_max = 1.05;
qnom_max_step_per_cycle = 0.03;

% ---------------------------------------------------------
% SELECTED-CYCLE TABLE EXPORT SETTINGS
% exactly to support the same style of plots as before
% ---------------------------------------------------------
save_selected_cycle_tables = true;
nGrid = 201;
tauGrid_plot = linspace(0,1,nGrid)';

late_window = [0.70 0.98];
mid_window  = [0.15 0.60];
phaseNames = {'FIRST','MIDDLE','LAST'};

%% =========================================================
% ADAPTIVE BAYES SETTINGS
%% =========================================================
P = struct();

% ---------- EKF uncertainty ----------
P.sigma_ekf_min = 5e-4;
P.sigma_ekf_max = 0.0300;
P.sigma_ekf_max_cap = 0.0600;

% ---------- EKF badness references ----------
P.innov_ref  = 10.0;
P.slope_ref  = 0.0500;
P.vresid_ref = 0.1000;

% EKF badness weights
P.w_innov  = 0.45;
P.w_slope  = 0.35;
P.w_vresid = 0.20;

% Mix instantaneous + cumulative evidence
P.w_innov_inst  = 0.65;
P.w_innov_mean  = 0.35;
P.w_vresid_inst = 0.50;
P.w_vresid_mean = 0.50;

% Plateau handling
P.plateau_slope_thresh = 0.040;
P.plateau_boost_gain   = 0.30;

% ---------- GPR uncertainty ----------
P.gpr_scale_base       = 1.80;
P.gpr_scale_min_factor = 0.60;
P.gpr_scale_max_factor = 1.20;
P.gpr_std_ref          = 0.0030;
P.gpr_std_floor        = 1e-4;
P.gpr_std_fallback     = 0.01;
P.sigma_gpr_cap        = 0.03;

% GPR badness components
P.w_gpr_std = 0.55;
P.w_dres    = 0.25;
P.w_flip    = 0.20;

P.dres_ref      = 0.0040;
P.sign_deadband = 5e-4;
P.flip_window   = 25;
P.flip_ref      = 0.10;

% Friendly regime shaping
P.friendly_sigma_ekf_boost  = 0.65;
P.friendly_gpr_scale_reduce = 0.45;
P.spike_sigma_gpr_gain      = 0.40;
P.flip_sigma_gpr_gain       = 0.35;

% ---------- causal smoothing ----------
P.alpha_ema       = 0.10;
P.alpha_ema_start = 0.55;
P.alpha_ema_end   = 0.18;

% ---------- residual smoothing ----------
P.residual_pred_ema_alpha = 0.45;

% ---------- memory from previous cycle ----------
P.mem_eta          = 0.20;
P.mem_mix_ekf      = 0.25;
P.mem_mix_gpr      = 0.20;
P.mem_init_ekf_bad = 0.55;
P.mem_init_gpr_bad = 0.20;
P.mem_init_alpha   = 0.28;

% ---------- startup release ----------
P.startup_prog_ref    = 0.18;
P.startup_alpha_floor = 0.60;
P.startup_alpha_boost = 0.55;
P.startup_raw_mix_max = 0.95;
P.startup_flip_relief = 1.00;
P.startup_dres_relief = 0.90;
P.startup_resid_ref   = 0.0025;

% ---------- dynamic alpha floor ----------
P.alpha_floor_base      = 0.05;
P.alpha_floor_friendly  = 0.22;
P.alpha_floor_startup   = 0.38;

%% =========================================================
% FOLDER SETUP
%% =========================================================
if save_plot_files && ~exist(plot_dir, 'dir')
    mkdir(plot_dir);
end

if (save_per_battery_csv || save_per_battery_mat) && ~exist(per_battery_dir, 'dir')
    mkdir(per_battery_dir);
end

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
        error('Missing core workspace vars and battery_workspace_core.mat not found.');
    end
end

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
% BUILD PRIMARY / SECONDARY SPLIT
%% =========================================================
all_batteries = 1:numel(t_all);

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

primary_test_batteries   = setdiff(primary_pool, exclude_batteries, 'stable');
secondary_pool           = setdiff(all_batteries, primary_pool, 'stable');
secondary_test_batteries = setdiff(secondary_pool, exclude_batteries, 'stable');

if use_secondary_default
    test_batteries = secondary_test_batteries(:)';
else
    test_batteries = intersect(custom_test_batteries(:)', all_batteries, 'stable');
end

if isempty(test_batteries)
    error('No test batteries selected.');
end

fprintf('\n=========================================================\n');
fprintf('ADAPTIVE BAYESIAN FUSION WITH PREVIOUS-CYCLE MEMORY\n');
fprintf('Testing batteries:\n%s\n', mat2str(test_batteries));
fprintf('Count = %d\n', numel(test_batteries));
fprintf('Plot requests:\n');
disp(plot_requests);
fprintf('=========================================================\n');

%% =========================================================
% MAIN LOOP
%% =========================================================
CycleResults = table();
BatterySummary = table();

SelRows = {};
TrajRows = {};

for ib = 1:numel(test_batteries)

    battery_no = test_batteries(ib);

    if battery_no > numel(t_all) || isempty(t_all{battery_no}) || ...
       battery_no > numel(I_all) || isempty(I_all{battery_no}) || ...
       battery_no > numel(V_all) || isempty(V_all{battery_no}) || ...
       battery_no > numel(Q_all) || isempty(Q_all{battery_no})
        fprintf('\nBattery %d skipped: missing data\n', battery_no);
        continue;
    end

    if battery_no > numel(Q_nom_init_per_battery) || ...
       ~isfinite(Q_nom_init_per_battery(battery_no)) || ...
       Q_nom_init_per_battery(battery_no) <= 0
        fprintf('\nBattery %d skipped: invalid Q_nom_init_per_battery\n', battery_no);
        continue;
    end

    Q_nom_init_batt = Q_nom_init_per_battery(battery_no) * 3600;
    Q_nom = Q_nom_init_batt;

    mem = init_cycle_memory(P);

    num_cycles = numel(t_all{battery_no});
    if num_cycles < 1
        continue;
    end

    % selected cycles for plot package
    selected_cycles = unique([1, round((1 + num_cycles)/2), num_cycles]);
    selected_cycles = selected_cycles(selected_cycles >= 1 & selected_cycles <= num_cycles);

    fprintf('\nBattery %d (%d/%d) | num_cycles = %d\n', ...
        battery_no, ib, numel(test_batteries), num_cycles);

    RowCell = cell(num_cycles,1);
    row_count = 0;

    rmse_ekf_all = [];
    rmse_gpr_all = [];
    rmse_bayes_all = [];

    rmse_ekf_late_all = [];
    rmse_gpr_late_all = [];
    rmse_bayes_late_all = [];

    bayes_improved_vs_ekf_all = [];
    bayes_better_than_gpr_all = [];
    bayes_worse_vs_ekf_all = [];
    bayes_worse_than_gpr_all = [];

    mean_alpha_bayes_all = [];
    mean_sigma_ekf_all = [];
    mean_sigma_gpr_all = [];
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

            [SOC_est_tmp, V_model_tmp, rmse_V_tmp, Q_accumulated, debug_tmp] = ...
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
                           numel(SOC_est_tmp), numel(V_model_tmp), ...
                           numel(debug_tmp.innovation), numel(debug_tmp.dOCV_dSOC), ...
                           numel(debug_tmp.Rk_eff), numel(debug_tmp.soc_gate_alpha)]);

            t = t_data(1:min_len);
            I_raw = I_data(1:min_len);
            V_raw = V_data(1:min_len);
            SOC_true = Q_series(1:min_len) / qmax;
            SOC_est = SOC_est_tmp(1:min_len);
            V_model = V_model_tmp(1:min_len);

            innovation = debug_tmp.innovation(1:min_len);
            dOCV_dSOC = debug_tmp.dOCV_dSOC(1:min_len);
            Rk_eff = debug_tmp.Rk_eff(1:min_len);
            soc_gate_alpha = debug_tmp.soc_gate_alpha(1:min_len);

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
            lateMask = progress_causal >= 0.80;

            Xtest = [ ...
                SOC_est, ...
                progress_causal, ...
                repmat(Q_nom_cycle_start / Q_nom_init_batt, numel(progress_causal), 1), ...
                v_resid_abs_mean_so_far, ...
                inv_abs_dOCV_dSOC_mean_so_far, ...
                norm_innov_mean_so_far, ...
                soc_gate_alpha_ema];

            valid_te = all(isfinite(Xtest), 2);

            residual_pred_raw = nan(size(SOC_est));
            residual_std_raw  = nan(size(SOC_est));

            if any(valid_te)
                [yp, ys] = predict_gpr_with_std(gpr_final_vC_lite, Xtest(valid_te,:));
                residual_pred_raw(valid_te) = yp;
                residual_std_raw(valid_te)  = ys;
            end

            residual_pred = max(min(residual_pred_raw, residual_clamp), -residual_clamp);

            residual_std = residual_std_raw;
            residual_std(~isfinite(residual_std)) = P.gpr_std_fallback;
            residual_std = max(residual_std, P.gpr_std_floor);

            residual_pred_f = smooth_residual_causal(residual_pred, P.residual_pred_ema_alpha);

            SOC_gpr = SOC_est + residual_pred;
            SOC_gpr = min(max(SOC_gpr, 0), 1);

            D = struct();
            D.battery_no = battery_no;
            D.cycle_idx = cycle_idx;
            D.t = t;
            D.I_raw = I_raw;
            D.V_raw = V_raw;
            D.SOC_true = SOC_true;
            D.SOC_est = SOC_est;
            D.SOC_gpr = SOC_gpr;
            D.V_model = V_model;

            D.residual_pred = residual_pred;
            D.residual_pred_f = residual_pred_f;
            D.residual_std = residual_std;

            D.abs_dOCV_dSOC = abs_dOCV_dSOC;
            D.abs_v_resid_inst = abs_v_resid;
            D.norm_innov_inst = norm_innov;

            D.norm_innov_mean_so_far = norm_innov_mean_so_far;
            D.v_resid_abs_mean_so_far = v_resid_abs_mean_so_far;

            D.progress_causal = progress_causal;
            D.lateMask = lateMask;
            D.rmse_v = rmse_V_tmp;
            D.qnom_start_frac = Q_nom_cycle_start / Q_nom_init_batt;
            D.qnom_was_clamped_target = qnom_was_clamped;

            M_ekf = evaluate_soc_method(D.SOC_est, D.SOC_true, D.lateMask);
            M_gpr = evaluate_soc_method(D.SOC_gpr, D.SOC_true, D.lateMask);
            [M_bayes, mem_out] = evaluate_adaptive_bayes_theta(D, mem, P);

            delta_gpr_vs_ekf   = M_gpr.rmse   - M_ekf.rmse;
            delta_bayes_vs_ekf = M_bayes.rmse - M_ekf.rmse;
            delta_bayes_vs_gpr = M_bayes.rmse - M_gpr.rmse;

            delta_gpr_vs_ekf_late   = M_gpr.rmse_late   - M_ekf.rmse_late;
            delta_bayes_vs_ekf_late = M_bayes.rmse_late - M_ekf.rmse_late;
            delta_bayes_vs_gpr_late = M_bayes.rmse_late - M_gpr.rmse_late;

            is_bayes_improved_vs_ekf = double(M_bayes.rmse < M_ekf.rmse);
            is_bayes_better_than_gpr = double(M_bayes.rmse < M_gpr.rmse);

            is_bayes_worse_vs_ekf   = double(M_bayes.rmse > M_ekf.rmse);
            is_bayes_worse_than_gpr = double(M_bayes.rmse > M_gpr.rmse);

            group_label = test_group_label(battery_no, primary_test_batteries, secondary_test_batteries);

            row_count = row_count + 1;
            RowCell{row_count} = table( ...
                categorical({group_label}, {'primary','secondary','custom'}), ...
                battery_no, cycle_idx, ...
                D.qnom_start_frac, ...
                qnom_was_clamped, ...
                M_ekf.rmse, M_gpr.rmse, M_bayes.rmse, ...
                delta_gpr_vs_ekf, delta_bayes_vs_ekf, delta_bayes_vs_gpr, ...
                M_ekf.rmse_late, M_gpr.rmse_late, M_bayes.rmse_late, ...
                delta_gpr_vs_ekf_late, delta_bayes_vs_ekf_late, delta_bayes_vs_gpr_late, ...
                mean(abs(residual_pred), 'omitnan'), ...
                mean(M_bayes.alpha, 'omitnan'), ...
                mean(M_bayes.sigma_ekf, 'omitnan'), ...
                mean(M_bayes.sigma_gpr, 'omitnan'), ...
                mean(M_bayes.gpr_scale_eff, 'omitnan'), ...
                mem.ekf_bad_bias, ...
                mem.gpr_bad_bias, ...
                D.rmse_v, ...
                is_bayes_improved_vs_ekf, ...
                is_bayes_better_than_gpr, ...
                is_bayes_worse_vs_ekf, ...
                is_bayes_worse_than_gpr, ...
                double(ismember(cycle_idx, selected_cycles)), ...
                'VariableNames', { ...
                'test_group','battery_no','cycle_idx', ...
                'qnom_start_frac','qnom_was_clamped', ...
                'rmse_ekf','rmse_gpr','rmse_bayes', ...
                'delta_gpr_vs_ekf','delta_bayes_vs_ekf','delta_bayes_vs_gpr', ...
                'rmse_ekf_late','rmse_gpr_late','rmse_bayes_late', ...
                'delta_gpr_vs_ekf_late','delta_bayes_vs_ekf_late','delta_bayes_vs_gpr_late', ...
                'mean_abs_residual_pred', ...
                'mean_alpha_bayes','mean_sigma_ekf','mean_sigma_gpr','mean_gpr_scale_eff', ...
                'mem_ekf_bad_in','mem_gpr_bad_in', ...
                'rmse_v', ...
                'is_bayes_improved_vs_ekf', ...
                'is_bayes_better_than_gpr', ...
                'is_bayes_worse_vs_ekf', ...
                'is_bayes_worse_than_gpr', ...
                'is_selected_cycle'});

            rmse_ekf_all = [rmse_ekf_all; M_ekf.rmse]; %#ok<AGROW>
            rmse_gpr_all = [rmse_gpr_all; M_gpr.rmse]; %#ok<AGROW>
            rmse_bayes_all = [rmse_bayes_all; M_bayes.rmse]; %#ok<AGROW>

            rmse_ekf_late_all = [rmse_ekf_late_all; M_ekf.rmse_late]; %#ok<AGROW>
            rmse_gpr_late_all = [rmse_gpr_late_all; M_gpr.rmse_late]; %#ok<AGROW>
            rmse_bayes_late_all = [rmse_bayes_late_all; M_bayes.rmse_late]; %#ok<AGROW>

            bayes_improved_vs_ekf_all = [bayes_improved_vs_ekf_all; is_bayes_improved_vs_ekf]; %#ok<AGROW>
            bayes_better_than_gpr_all = [bayes_better_than_gpr_all; is_bayes_better_than_gpr]; %#ok<AGROW>
            bayes_worse_vs_ekf_all    = [bayes_worse_vs_ekf_all; is_bayes_worse_vs_ekf]; %#ok<AGROW>
            bayes_worse_than_gpr_all  = [bayes_worse_than_gpr_all; is_bayes_worse_than_gpr]; %#ok<AGROW>

            mean_alpha_bayes_all = [mean_alpha_bayes_all; mean(M_bayes.alpha,'omitnan')]; %#ok<AGROW>
            mean_sigma_ekf_all = [mean_sigma_ekf_all; mean(M_bayes.sigma_ekf,'omitnan')]; %#ok<AGROW>
            mean_sigma_gpr_all = [mean_sigma_gpr_all; mean(M_bayes.sigma_gpr,'omitnan')]; %#ok<AGROW>
            qnom_clamped_all = [qnom_clamped_all; qnom_was_clamped]; %#ok<AGROW>

            % -----------------------------------------------------
            % PLOT PACKAGE TABLES
            % -----------------------------------------------------
            if save_selected_cycle_tables && ismember(cycle_idx, selected_cycles)

                abs_err_ekf = abs(SOC_true - SOC_est);
                abs_err_gpr = abs(SOC_true - SOC_gpr);
                abs_err_bay = abs(SOC_true - M_bayes.SOC);

                res_energy_ekf = trapz(build_tau_plot(t), (SOC_true - SOC_est).^2);
                res_energy_gpr = trapz(build_tau_plot(t), (SOC_true - SOC_gpr).^2);
                res_energy_bay = trapz(build_tau_plot(t), (SOC_true - M_bayes.SOC).^2);

                tau_plot = build_tau_plot(t);

                [late_peak_ekf, late_t_ekf] = extract_peak_feature(tau_plot, SOC_true - SOC_est, late_window, 'max');
                [late_peak_gpr, late_t_gpr] = extract_peak_feature(tau_plot, SOC_true - SOC_gpr, late_window, 'max');
                [late_peak_bay, late_t_bay] = extract_peak_feature(tau_plot, SOC_true - M_bayes.SOC, late_window, 'max');

                [mid_valley_ekf, mid_t_ekf] = extract_peak_feature(tau_plot, SOC_true - SOC_est, mid_window, 'min');
                [mid_valley_gpr, mid_t_gpr] = extract_peak_feature(tau_plot, SOC_true - SOC_gpr, mid_window, 'min');
                [mid_valley_bay, mid_t_bay] = extract_peak_feature(tau_plot, SOC_true - M_bayes.SOC, mid_window, 'min');

                pidx = find(selected_cycles == cycle_idx, 1, 'first');
                if numel(selected_cycles) == 1
                    phase_label = 'MIDDLE';
                elseif numel(selected_cycles) == 2
                    phase_label = ternary(pidx == 1, 'FIRST', 'LAST');
                else
                    phase_label = phaseNames{pidx};
                end

                resid_ekf_i = interp1_monotonic_safe(tau_plot, SOC_true - SOC_est, tauGrid_plot);
                resid_gpr_i = interp1_monotonic_safe(tau_plot, SOC_true - SOC_gpr, tauGrid_plot);
                resid_bay_i = interp1_monotonic_safe(tau_plot, SOC_true - M_bayes.SOC, tauGrid_plot);

                abs_err_ekf_i = interp1_monotonic_safe(tau_plot, abs_err_ekf, tauGrid_plot);
                abs_err_gpr_i = interp1_monotonic_safe(tau_plot, abs_err_gpr, tauGrid_plot);
                abs_err_bay_i = interp1_monotonic_safe(tau_plot, abs_err_bay, tauGrid_plot);

                soc_true_i = interp1_monotonic_safe(tau_plot, SOC_true, tauGrid_plot);
                soc_ekf_i  = interp1_monotonic_safe(tau_plot, SOC_est, tauGrid_plot);
                soc_gpr_i  = interp1_monotonic_safe(tau_plot, SOC_gpr, tauGrid_plot);
                soc_bay_i  = interp1_monotonic_safe(tau_plot, M_bayes.SOC, tauGrid_plot);

                v_true_i   = interp1_monotonic_safe(tau_plot, V_raw, tauGrid_plot);
                v_model_i  = interp1_monotonic_safe(tau_plot, V_model, tauGrid_plot);

                SelRows(end+1,1) = {table( ...
                    battery_no, cycle_idx, string(phase_label), ...
                    D.qnom_start_frac, ...
                    qnom_was_clamped, ...
                    M_ekf.rmse, M_gpr.rmse, M_bayes.rmse, ...
                    delta_gpr_vs_ekf, delta_bayes_vs_ekf, delta_bayes_vs_gpr, ...
                    res_energy_ekf, res_energy_gpr, res_energy_bay, ...
                    late_peak_ekf, late_t_ekf, ...
                    late_peak_gpr, late_t_gpr, ...
                    late_peak_bay, late_t_bay, ...
                    mid_valley_ekf, mid_t_ekf, ...
                    mid_valley_gpr, mid_t_gpr, ...
                    mid_valley_bay, mid_t_bay, ...
                    D.rmse_v, ...
                    'VariableNames', { ...
                    'battery_no','cycle_idx','phase', ...
                    'soh_proxy','qnom_was_clamped', ...
                    'rmse_ekf','rmse_gpr','rmse_bayes', ...
                    'delta_gpr_vs_ekf','delta_bayes_vs_ekf','delta_bayes_vs_gpr', ...
                    'res_energy_ekf','res_energy_gpr','res_energy_bayes', ...
                    'late_peak_ekf','late_t_ekf', ...
                    'late_peak_gpr','late_t_gpr', ...
                    'late_peak_bayes','late_t_bayes', ...
                    'mid_valley_ekf','mid_t_ekf', ...
                    'mid_valley_gpr','mid_t_gpr', ...
                    'mid_valley_bayes','mid_t_bayes', ...
                    'rmse_v'})};

                TrajRows(end+1,1) = {table( ...
                    repmat(battery_no, nGrid, 1), ...
                    repmat(cycle_idx, nGrid, 1), ...
                    repmat(string(phase_label), nGrid, 1), ...
                    tauGrid_plot, ...
                    soc_true_i, ...
                    soc_ekf_i, ...
                    soc_gpr_i, ...
                    soc_bay_i, ...
                    v_true_i, ...
                    v_model_i, ...
                    resid_ekf_i, ...
                    resid_gpr_i, ...
                    resid_bay_i, ...
                    abs_err_ekf_i, ...
                    abs_err_gpr_i, ...
                    abs_err_bay_i, ...
                    'VariableNames', { ...
                    'battery_no','cycle_idx','phase','tau', ...
                    'soc_true','soc_ekf','soc_gpr','soc_bayes', ...
                    'v_true','v_model', ...
                    'resid_ekf','resid_gpr','resid_bayes', ...
                    'abs_err_ekf','abs_err_gpr','abs_err_bayes'})};
            end

            if make_requested_plots && matches_plot_request(battery_no, cycle_idx, plot_requests)
                make_cycle_diagnostic_plot(D, M_ekf, M_gpr, M_bayes, mem, mem_out, ...
                    plot_mode, save_plot_files, plot_dir);
            end

            mem = mem_out;
            Q_nom = Q_nom_next;

            if cycle_idx == 1 || cycle_idx == num_cycles || ...
               mod(cycle_idx, print_every_n_cycles) == 0 || ...
               matches_plot_request(battery_no, cycle_idx, plot_requests)

                fprintf(['  Cycle %4d | EKF = %.6f | GPR = %.6f | ' ...
                         'Bayes = %.6f | Bayes-EKF = %.6f | alpha = %.4f\n'], ...
                    cycle_idx, M_ekf.rmse, M_gpr.rmse, M_bayes.rmse, ...
                    M_bayes.rmse - M_ekf.rmse, mean(M_bayes.alpha,'omitnan'));
            end

        catch ME
            fprintf('  Battery %d | cycle %d skipped: %s\n', battery_no, cycle_idx, ME.message);
            continue;
        end
    end

    BatteryCycleTable = table();
    if row_count > 0
        BatteryCycleTable = vertcat(RowCell{1:row_count});
        BatteryCycleTable = sortrows(BatteryCycleTable, {'battery_no','cycle_idx'}, {'ascend','ascend'});
        CycleResults = [CycleResults; BatteryCycleTable]; %#ok<AGROW>
    end

    BatterySummaryRow = table();
    if ~isempty(rmse_ekf_all)
        group_label = test_group_label(battery_no, primary_test_batteries, secondary_test_batteries);

        BatterySummaryRow = table( ...
            categorical({group_label}, {'primary','secondary','custom'}), ...
            battery_no, ...
            numel(rmse_ekf_all), ...
            mean(rmse_ekf_all,'omitnan'), ...
            mean(rmse_gpr_all,'omitnan'), ...
            mean(rmse_bayes_all,'omitnan'), ...
            mean(rmse_ekf_late_all,'omitnan'), ...
            mean(rmse_gpr_late_all,'omitnan'), ...
            mean(rmse_bayes_late_all,'omitnan'), ...
            sum(bayes_improved_vs_ekf_all,'omitnan'), ...
            sum(bayes_better_than_gpr_all,'omitnan'), ...
            sum(bayes_worse_vs_ekf_all,'omitnan'), ...
            sum(bayes_worse_than_gpr_all,'omitnan'), ...
            mean(bayes_improved_vs_ekf_all,'omitnan'), ...
            mean(bayes_better_than_gpr_all,'omitnan'), ...
            mean(mean_alpha_bayes_all,'omitnan'), ...
            mean(mean_sigma_ekf_all,'omitnan'), ...
            mean(mean_sigma_gpr_all,'omitnan'), ...
            sum(qnom_clamped_all,'omitnan'), ...
            'VariableNames', { ...
            'test_group','battery_no','num_cycles_evaluated', ...
            'mean_rmse_ekf','mean_rmse_gpr','mean_rmse_bayes', ...
            'mean_rmse_ekf_late','mean_rmse_gpr_late','mean_rmse_bayes_late', ...
            'num_bayes_improved_vs_ekf', ...
            'num_bayes_better_than_gpr', ...
            'num_bayes_worse_vs_ekf', ...
            'num_bayes_worse_than_gpr', ...
            'bayes_improve_rate_vs_ekf', ...
            'bayes_better_rate_vs_gpr', ...
            'mean_alpha_bayes','mean_sigma_ekf','mean_sigma_gpr', ...
            'num_qnom_clamped_cycles'});

        BatterySummary = [BatterySummary; BatterySummaryRow]; %#ok<AGROW>
    end

    if save_per_battery_csv && ~isempty(BatteryCycleTable)
        writetable(BatteryCycleTable, fullfile(per_battery_dir, sprintf('battery_%03d_cycle_results.csv', battery_no)));
        if ~isempty(BatterySummaryRow)
            writetable(BatterySummaryRow, fullfile(per_battery_dir, sprintf('battery_%03d_summary.csv', battery_no)));
        end
    end

    if save_per_battery_mat
        save(fullfile(per_battery_dir, sprintf('battery_%03d_outputs.mat', battery_no)), ...
            'battery_no', 'BatteryCycleTable', 'BatterySummaryRow', 'P');
    end

    if save_partial_mat
        save(partial_outputs_mat, ...
            'CycleResults', 'BatterySummary', ...
            'test_batteries', ...
            'primary_test_batteries', 'secondary_test_batteries', ...
            'train_batteries_eval', 'exclude_batteries', ...
            'plot_requests', 'P', ...
            'qnom_ratio_min', 'qnom_ratio_max', 'qnom_max_step_per_cycle');
    end
end

%% =========================================================
% SORT + PLOT PACKAGE TABLES
%% =========================================================
if ~isempty(CycleResults) && istable(CycleResults) && ...
   ismember('battery_no', CycleResults.Properties.VariableNames) && ...
   ismember('cycle_idx', CycleResults.Properties.VariableNames)
    CycleResults = sortrows(CycleResults, {'battery_no','cycle_idx'}, {'ascend','ascend'});
end

if ~isempty(BatterySummary) && istable(BatterySummary) && ...
   ismember('battery_no', BatterySummary.Properties.VariableNames)
    BatterySummary = sortrows(BatterySummary, 'battery_no', 'ascend');
end

if ~isempty(SelRows)
    BayesSelTable = vertcat(SelRows{:});
    BayesSelTable.phase = categorical(string(BayesSelTable.phase), phaseNames, 'Ordinal', true);
else
    BayesSelTable = table();
end

if ~isempty(TrajRows)
    BayesTrajTable = vertcat(TrajRows{:});
    BayesTrajTable.phase = categorical(string(BayesTrajTable.phase), phaseNames, 'Ordinal', true);
else
    BayesTrajTable = table();
end

%% =========================================================
% DISPLAY
%% =========================================================
disp(' ');
disp('==================== ADAPTIVE BAYES CYCLE RESULTS (HEAD) ====================');
if ~isempty(CycleResults)
    disp(CycleResults(1:min(20,height(CycleResults)), :));
else
    disp('CycleResults is empty.');
end

disp(' ');
disp('==================== ADAPTIVE BAYES BATTERY SUMMARY ====================');
if ~isempty(BatterySummary)
    disp(BatterySummary);
else
    disp('BatterySummary is empty.');
end

disp(' ');
disp('==================== BAYES SELECTED-CYCLE TABLE (HEAD) ====================');
if ~isempty(BayesSelTable)
    disp(BayesSelTable(1:min(10,height(BayesSelTable)), :));
else
    disp('BayesSelTable is empty.');
end

if ~isempty(BatterySummary)
    fprintf('\n---------------- OVERALL TEST SUMMARY ----------------\n');
    fprintf('Mean EKF RMSE               = %.6f\n', mean(BatterySummary.mean_rmse_ekf,'omitnan'));
    fprintf('Mean GPR RMSE               = %.6f\n', mean(BatterySummary.mean_rmse_gpr,'omitnan'));
    fprintf('Mean Adaptive Bayes RMSE    = %.6f\n', mean(BatterySummary.mean_rmse_bayes,'omitnan'));

    fprintf('Mean EKF late RMSE          = %.6f\n', mean(BatterySummary.mean_rmse_ekf_late,'omitnan'));
    fprintf('Mean GPR late RMSE          = %.6f\n', mean(BatterySummary.mean_rmse_gpr_late,'omitnan'));
    fprintf('Mean Adaptive Bayes late    = %.6f\n', mean(BatterySummary.mean_rmse_bayes_late,'omitnan'));

    fprintf('Mean Bayes improve rate vs EKF = %.2f %%\n', ...
        100 * mean(BatterySummary.bayes_improve_rate_vs_ekf,'omitnan'));
    fprintf('Mean Bayes better rate vs GPR  = %.2f %%\n', ...
        100 * mean(BatterySummary.bayes_better_rate_vs_gpr,'omitnan'));
    fprintf('Total Bayes worse than EKF     = %d\n', ...
        sum(BatterySummary.num_bayes_worse_vs_ekf,'omitnan'));
    fprintf('Total Q_nom clamped cycles     = %d\n', ...
        sum(BatterySummary.num_qnom_clamped_cycles,'omitnan'));
end

%% =========================================================
% SAVE
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

    fprintf('\nSaved CSVs:\n');
    fprintf('  %s\n', cycle_csv_name);
    fprintf('  %s\n', summary_csv_name);
end

if save_mat
    save(outputs_mat_name, ...
        'CycleResults', 'BatterySummary', ...
        'test_batteries', ...
        'primary_test_batteries', 'secondary_test_batteries', ...
        'train_batteries_eval', 'exclude_batteries', ...
        'plot_requests', 'P', ...
        'qnom_ratio_min', 'qnom_ratio_max', 'qnom_max_step_per_cycle');

    fprintf('Saved MAT:\n');
    fprintf('  %s\n', outputs_mat_name);
end

if save_plot_package
    save(plot_package_mat_name, ...
        'CycleResults', 'BatterySummary', ...
        'BayesSelTable', 'BayesTrajTable', ...
        'test_batteries', ...
        'primary_test_batteries', 'secondary_test_batteries', ...
        'train_batteries_eval', 'exclude_batteries', ...
        'plot_requests', 'P', ...
        'qnom_ratio_min', 'qnom_ratio_max', 'qnom_max_step_per_cycle', ...
        'residual_clamp', 'min_cycle_length', ...
        '-v7.3');

    fprintf('Saved plot package MAT:\n');
    fprintf('  %s\n', plot_package_mat_name);
end

%% =========================================================
% LOCAL FUNCTIONS
%% =========================================================
function mem = init_cycle_memory(P)
    mem = struct();
    mem.ekf_bad_bias = P.mem_init_ekf_bad;
    mem.gpr_bad_bias = P.mem_init_gpr_bad;
    mem.alpha_prev   = P.mem_init_alpha;
end

function label = test_group_label(battery_no, primary_set, secondary_set)
    if ismember(battery_no, primary_set)
        label = 'primary';
    elseif ismember(battery_no, secondary_set)
        label = 'secondary';
    else
        label = 'custom';
    end
end

function tf = matches_plot_request(battery_no, cycle_idx, plot_requests)
    tf = false;
    if isempty(plot_requests)
        return;
    end
    tf = any(plot_requests(:,1) == battery_no & plot_requests(:,2) == cycle_idx);
end

function [y_pred, y_std] = predict_gpr_with_std(model, X)
    try
        [y_pred, y_std] = predict(model, X);
    catch
        y_pred = predict(model, X);

        y_std = nan(size(y_pred));
        if isprop(model, 'Sigma')
            y_std(:) = model.Sigma;
        elseif isprop(model, 'ResidualSigma')
            y_std(:) = model.ResidualSigma;
        else
            y_std(:) = 0.01;
        end
    end
end

function M = evaluate_soc_method(SOC_hat, SOC_true, lateMask)
    e = SOC_hat - SOC_true;
    M = struct();
    M.SOC = SOC_hat;
    M.rmse = sqrt(mean(e.^2, 'omitnan'));

    if any(lateMask)
        M.rmse_late = sqrt(mean(e(lateMask).^2, 'omitnan'));
    else
        M.rmse_late = NaN;
    end
end

function [M, mem_out] = evaluate_adaptive_bayes_theta(D, mem_in, P)

    startup = max(0, 1 - D.progress_causal / max(P.startup_prog_ref, 1e-12));
    startup = min(max(startup, 0), 1);

    slope_conf = min(max(D.abs_dOCV_dSOC ./ max(P.slope_ref,1e-12), 0), 1);
    slope_bad  = 1 - slope_conf;

    innov_bad_inst = min(max(D.norm_innov_inst ./ max(P.innov_ref,1e-12), 0), 1);
    innov_bad_mean = min(max(D.norm_innov_mean_so_far ./ max(P.innov_ref,1e-12), 0), 1);
    innov_bad = P.w_innov_inst * innov_bad_inst + P.w_innov_mean * innov_bad_mean;

    vresid_bad_inst = min(max(D.abs_v_resid_inst ./ max(P.vresid_ref,1e-12), 0), 1);
    vresid_bad_mean = min(max(D.v_resid_abs_mean_so_far ./ max(P.vresid_ref,1e-12), 0), 1);
    vresid_bad = P.w_vresid_inst * vresid_bad_inst + P.w_vresid_mean * vresid_bad_mean;

    ekf_bad_curr = P.w_innov * innov_bad + ...
                   P.w_slope * slope_bad + ...
                   P.w_vresid * vresid_bad;

    plateau_boost = double(D.abs_dOCV_dSOC < P.plateau_slope_thresh) .* innov_bad_inst;
    ekf_bad_curr = ekf_bad_curr + P.plateau_boost_gain * plateau_boost;
    ekf_bad_curr = min(max(ekf_bad_curr, 0), 1);

    gpr_std_bad = min(max(D.residual_std ./ max(P.gpr_std_ref,1e-12), 0), 1);

    dres = [0; abs(diff(D.residual_pred_f))];
    dres_bad = min(max(dres ./ max(P.dres_ref,1e-12), 0), 1);

    s = sign(D.residual_pred_f);
    s(abs(D.residual_pred_f) < P.sign_deadband) = 0;
    flip_evt = [0; double(s(2:end) ~= s(1:end-1))];
    flip_rate = movmean(flip_evt, min(P.flip_window, numel(flip_evt)));
    flip_bad = min(max(flip_rate ./ max(P.flip_ref,1e-12), 0), 1);

    dres_relief = 1 - P.startup_dres_relief * startup;
    flip_relief = 1 - P.startup_flip_relief * startup;

    gpr_bad_curr = P.w_gpr_std * gpr_std_bad + ...
                   P.w_dres    * (dres_relief .* dres_bad) + ...
                   P.w_flip    * (flip_relief .* flip_bad);
    gpr_bad_curr = min(max(gpr_bad_curr, 0), 1);

    ekf_bad = (1 - P.mem_mix_ekf) * ekf_bad_curr + P.mem_mix_ekf * mem_in.ekf_bad_bias;
    gpr_bad = (1 - P.mem_mix_gpr) * gpr_bad_curr + P.mem_mix_gpr * mem_in.gpr_bad_bias;

    ekf_bad = min(max(ekf_bad, 0), 1);
    gpr_bad = min(max(gpr_bad, 0), 1);

    friendly = min(max((1 - gpr_bad) .* ekf_bad, 0), 1);

    sigma_ekf_max_eff = P.sigma_ekf_max .* (1 + P.friendly_sigma_ekf_boost .* friendly);
    sigma_ekf_max_eff = min(sigma_ekf_max_eff, P.sigma_ekf_max_cap);

    sigma_ekf = P.sigma_ekf_min + (sigma_ekf_max_eff - P.sigma_ekf_min) .* ekf_bad;

    startup_conf = startup .* (1 - gpr_bad) .* ...
        min(max(abs(D.residual_pred_f) ./ max(P.startup_resid_ref,1e-12), 0), 1);

    sigma_ekf = sigma_ekf .* (1 + 0.9 * startup_conf);

    sigma_ekf(~isfinite(sigma_ekf)) = P.sigma_ekf_min;
    sigma_ekf = max(sigma_ekf, P.sigma_ekf_min);

    gpr_scale_eff = P.gpr_scale_base .* ...
        (1 + (P.gpr_scale_max_factor - 1) .* gpr_bad - P.friendly_gpr_scale_reduce .* friendly);

    gpr_scale_eff = max(gpr_scale_eff, P.gpr_scale_base * P.gpr_scale_min_factor);
    gpr_scale_eff = min(gpr_scale_eff, P.gpr_scale_base * P.gpr_scale_max_factor);

    sigma_gpr = gpr_scale_eff .* D.residual_std .* ...
        (1 + P.spike_sigma_gpr_gain .* dres_bad + P.flip_sigma_gpr_gain .* flip_bad);

    sigma_gpr(~isfinite(sigma_gpr)) = P.gpr_std_fallback;
    sigma_gpr = max(sigma_gpr, P.gpr_std_floor);
    sigma_gpr = min(sigma_gpr, P.sigma_gpr_cap);

    alpha_raw = sigma_ekf.^2 ./ (sigma_ekf.^2 + sigma_gpr.^2);
    alpha_raw(~isfinite(alpha_raw)) = 0.5;
    alpha_raw = min(max(alpha_raw, 0), 1);

    alpha_raw = alpha_raw + P.startup_alpha_boost .* startup_conf .* (1 - alpha_raw);
    alpha_raw = max(alpha_raw, P.startup_alpha_floor .* startup_conf);
    alpha_raw = min(max(alpha_raw, 0), 1);

    alpha_ema_eff = P.alpha_ema_end + (P.alpha_ema_start - P.alpha_ema_end) .* startup;
    alpha_bayes = causalEMA_init_variable(alpha_raw, alpha_ema_eff, mem_in.alpha_prev);
    alpha_bayes = min(max(alpha_bayes, 0), 1);

    raw_mix = P.startup_raw_mix_max .* startup_conf;
    residual_for_fusion = raw_mix .* D.residual_pred + (1 - raw_mix) .* D.residual_pred_f;

    residual_bayes = alpha_bayes .* residual_for_fusion;
    SOC_bayes = D.SOC_est + residual_bayes;
    SOC_bayes = min(max(SOC_bayes, 0), 1);

    e = SOC_bayes - D.SOC_true;

    M = struct();
    M.SOC = SOC_bayes;
    M.alpha = alpha_bayes;
    M.alpha_raw = alpha_raw;
    M.sigma_ekf = sigma_ekf;
    M.sigma_gpr = sigma_gpr;
    M.gpr_scale_eff = gpr_scale_eff;
    M.ekf_bad = ekf_bad;
    M.gpr_bad = gpr_bad;
    M.friendly = friendly;
    M.startup = startup;
    M.startup_conf = startup_conf;
    M.residual_bayes = residual_bayes;
    M.residual_for_fusion = residual_for_fusion;
    M.rmse = sqrt(mean(e.^2, 'omitnan'));

    if any(D.lateMask)
        M.rmse_late = sqrt(mean(e(D.lateMask).^2, 'omitnan'));
    else
        M.rmse_late = NaN;
    end

    mem_out = mem_in;

    ekf_bad_cycle = mean(ekf_bad_curr, 'omitnan');
    gpr_bad_cycle = mean(gpr_bad_curr, 'omitnan');

    if ~isfinite(ekf_bad_cycle), ekf_bad_cycle = mem_in.ekf_bad_bias; end
    if ~isfinite(gpr_bad_cycle), gpr_bad_cycle = mem_in.gpr_bad_bias; end

    mem_out.ekf_bad_bias = (1 - P.mem_eta) * mem_in.ekf_bad_bias + P.mem_eta * ekf_bad_cycle;
    mem_out.gpr_bad_bias = (1 - P.mem_eta) * mem_in.gpr_bad_bias + P.mem_eta * gpr_bad_cycle;

    tailN = min(25, numel(alpha_bayes));
    alpha_tail = alpha_bayes(end-tailN+1:end);
    alpha_tail = alpha_tail(isfinite(alpha_tail));

    if isempty(alpha_tail)
        mem_out.alpha_prev = mem_in.alpha_prev;
    else
        mem_out.alpha_prev = mean(alpha_tail);
    end
end

function make_cycle_diagnostic_plot(D, M_ekf, M_gpr, M_bayes, mem_in, mem_out, plot_mode, save_plot_files, plot_dir)

    batt = D.battery_no;
    cyc  = D.cycle_idx;

    if strcmpi(plot_mode, 'residual_only')
        f = figure('Color','w','Position',[120 120 1200 500]);
        tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

        nexttile; hold on;
        plot(D.t, D.SOC_true - D.SOC_est, 'k-', 'LineWidth', 1.5, 'DisplayName', 'True residual');
        plot(D.t, D.residual_pred, 'm--', 'LineWidth', 1.2, 'DisplayName', 'Pred residual raw');
        plot(D.t, D.residual_pred_f, 'c-', 'LineWidth', 1.2, 'DisplayName', 'Pred residual filt');
        plot(D.t, M_bayes.residual_bayes, 'r-', 'LineWidth', 1.4, 'DisplayName', 'Applied residual');
        yline(0,'k:');
        xlabel('Time');
        ylabel('Residual');
        title(sprintf('Residuals | Batt %d Cycle %d', batt, cyc));
        legend('Location','best');
        grid on;

        nexttile; hold on;
        plot(D.t, M_bayes.alpha, 'b-', 'LineWidth', 1.4, 'DisplayName', '\alpha bayes');
        plot(D.t, M_bayes.sigma_ekf, 'r-', 'LineWidth', 1.2, 'DisplayName', '\sigma_{EKF}');
        plot(D.t, M_bayes.sigma_gpr, 'm-', 'LineWidth', 1.2, 'DisplayName', '\sigma_{GPR}');
        xlabel('Time');
        ylabel('Weight / Uncertainty');
        title(sprintf('Weighting | mem in = [%.3f %.3f] -> out = [%.3f %.3f]', ...
            mem_in.ekf_bad_bias, mem_in.gpr_bad_bias, mem_out.ekf_bad_bias, mem_out.gpr_bad_bias));
        legend('Location','best');
        grid on;

    else
        f = figure('Color','w','Position',[80 80 1400 900]);
        tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

        nexttile; hold on;
        plot(D.t, D.SOC_true, 'k-', 'LineWidth', 1.8, 'DisplayName', 'True SOC');
        plot(D.t, D.SOC_est, 'b--', 'LineWidth', 1.2, 'DisplayName', 'EKF');
        plot(D.t, D.SOC_gpr, 'm-', 'LineWidth', 1.2, 'DisplayName', 'Raw GPR');
        plot(D.t, M_bayes.SOC, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Adaptive Bayes');
        xlabel('Time');
        ylabel('SOC');
        title(sprintf('SOC | Batt %d Cycle %d', batt, cyc));
        legend('Location','best');
        grid on;

        nexttile; hold on;
        plot(D.t, D.SOC_true - D.SOC_est, 'k-', 'LineWidth', 1.4, 'DisplayName', 'True residual');
        plot(D.t, D.residual_pred, 'm--', 'LineWidth', 1.1, 'DisplayName', 'Pred residual raw');
        plot(D.t, D.residual_pred_f, 'c-', 'LineWidth', 1.2, 'DisplayName', 'Pred residual filt');
        plot(D.t, M_bayes.residual_bayes, 'r-', 'LineWidth', 1.3, 'DisplayName', 'Applied residual');
        yline(0,'k:');
        xlabel('Time');
        ylabel('Residual');
        title('Residual correction');
        legend('Location','best');
        grid on;

        nexttile; hold on;
        plot(D.t, M_bayes.alpha, 'b-', 'LineWidth', 1.4, 'DisplayName', '\alpha bayes');
        plot(D.t, M_bayes.sigma_ekf, 'r-', 'LineWidth', 1.2, 'DisplayName', '\sigma_{EKF}');
        plot(D.t, M_bayes.sigma_gpr, 'm-', 'LineWidth', 1.2, 'DisplayName', '\sigma_{GPR}');
        xlabel('Time');
        ylabel('Weight / Uncertainty');
        title('Adaptive uncertainty weighting');
        legend('Location','best');
        grid on;

        nexttile; axis off;
        txt = {
            sprintf('Battery = %d', batt)
            sprintf('Cycle   = %d', cyc)
            ' '
            sprintf('EKF RMSE         = %.6f', M_ekf.rmse)
            sprintf('Raw GPR RMSE     = %.6f', M_gpr.rmse)
            sprintf('Adaptive Bayes   = %.6f', M_bayes.rmse)
            ' '
            sprintf('EKF late         = %.6f', M_ekf.rmse_late)
            sprintf('GPR late         = %.6f', M_gpr.rmse_late)
            sprintf('Bayes late       = %.6f', M_bayes.rmse_late)
            ' '
            sprintf('mem in  = [ekf %.3f | gpr %.3f | alpha %.3f]', ...
                mem_in.ekf_bad_bias, mem_in.gpr_bad_bias, mem_in.alpha_prev)
            sprintf('mem out = [ekf %.3f | gpr %.3f | alpha %.3f]', ...
                mem_out.ekf_bad_bias, mem_out.gpr_bad_bias, mem_out.alpha_prev)
            };
        text(0.02, 0.98, txt, 'VerticalAlignment','top', 'FontName','Consolas', 'FontSize', 11);
    end

    if save_plot_files
        exportgraphics(f, fullfile(plot_dir, sprintf('battery_%03d_cycle_%04d_%s.png', batt, cyc, lower(plot_mode))), 'Resolution', 250);
    end
end

function y = smooth_residual_causal(x, alpha)
    x = x(:);
    y = nan(size(x));

    if isempty(x)
        return;
    end

    first = find(isfinite(x), 1, 'first');
    if isempty(first)
        y = zeros(size(x));
        return;
    end

    y(1:first) = x(first);

    for k = first+1:numel(x)
        if isfinite(x(k))
            y(k) = alpha * x(k) + (1 - alpha) * y(k-1);
        else
            y(k) = y(k-1);
        end
    end

    y(~isfinite(y)) = 0;
end

function y = causalEMA(x, alpha)
    x = x(:);
    y = zeros(size(x));
    if isempty(x), return; end
    y(1) = x(1);
    for k = 2:numel(x)
        y(k) = alpha * x(k) + (1 - alpha) * y(k-1);
    end
end

function y = causalEMA_init_variable(x, alpha_vec, y0)
    x = x(:);
    alpha_vec = alpha_vec(:);
    y = zeros(size(x));

    if isempty(x)
        return;
    end

    if ~isfinite(y0)
        y0 = x(1);
    end

    a1 = alpha_vec(1);
    y(1) = a1 * x(1) + (1 - a1) * y0;

    for k = 2:numel(x)
        a = alpha_vec(k);
        y(k) = a * x(k) + (1 - a) * y(k-1);
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