%% =========================================================
% STAGE 2 (SINGLE-MODEL): SAMPLE-LEVEL FEATURE SCREENING
% WITH BATTERY-SPECIFIC Q_nom_init_per_battery LOADED FROM MAT
%
% GOAL:
%   Build one global sample-level table and rank deployable
%   features for a single residual GPR model.
%
% INPUT FILES:
%   battery_workspace_core.mat
%   fusion_full_model.mat
%   Q_nom_init_first_cycle_all_batteries.mat      -> must contain variable: Q_nom_init_per_battery
%
% OUTPUTS:
%   stage2_feature_scores_single.csv
%   stage2_feature_top_pool_single.csv
%% =========================================================

% clear; clc; close all;
% rng(42);

%% =========================================================
% LOAD
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

if exist('fusion_full_model.mat', 'file') == 2
    load('fusion_full_model.mat');
else
    error('Missing required file: fusion_full_model.mat.');
end

if exist('Q_nom_init_first_cycle_all_batteries.mat', 'file') == 2
    load('Q_nom_init_first_cycle_all_batteries.mat', 'Q_nom_init_per_battery');
else
    error('Missing required file: Q_nom_init_first_cycle_all_batteries.mat.');
end

%% =========================================================
% CHECKS
%% =========================================================
requiredVars = { ...
    't_all','I_all','V_all','Q_all', ...
    'train_batteries', ...
    'I_noise_std','V_noise_std', ...
    'R0','R1','C1','R2','C2', ...
    'Q_nom_init_per_battery'};

for k = 1:numel(requiredVars)
    if exist(requiredVars{k}, 'var') ~= 1
        error('Missing variable: %s', requiredVars{k});
    end
end

if ~isvector(Q_nom_init_per_battery)
    error('Q_nom_init_per_battery must be a vector.');
end

Q_nom_init_per_battery = Q_nom_init_per_battery(:);

if numel(Q_nom_init_per_battery) < numel(t_all)
    error('Q_nom_init_per_battery has fewer entries than the available batteries.');
end

%% =========================================================
% SETTINGS
%% =========================================================
min_cycle_length = 30;

% -------- dense global sampling design --------
cycle_stride_train = 4;
base_step = 35;
n_region1 = 18;
n_region2 = 24;
n_region3 = 28;
n_high_innov = 10;

region1 = [0.00 0.10];
region2 = [0.20 0.45];
region3 = [0.78 0.95];

% extra dense early-cycle coverage, now used globally
use_extra_dense_early_mesh = true;

% -------- remove flagged training outlier cycles --------
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

% -------- candidate feature pool (deployment-safe) --------
candidateFeatures = { ...
    'soc_est'
    'tau'
    'innovation'
    'abs_innov'
    'innov_abs_ema'
    'norm_innov'
    'norm_innov_ema'
    'abs_dOCV_dSOC'
    'abs_dOCV_dSOC_ema'
    'throughput_frac'
    'soc_gate_alpha_ema'
    'v_resid_abs_mean_so_far'
    'inv_abs_dOCV_dSOC_mean_so_far'
    'norm_innov_mean_so_far'
    };

use_qnom_start_frac = true;
if use_qnom_start_frac
    candidateFeatures = [candidateFeatures; {'qnom_start_frac'}];
end

%% =========================================================
% OUTPUT NAMES
%% =========================================================
scores_csv  = 'stage2_feature_scores_single.csv';
top_csv     = 'stage2_feature_top_pool_single.csv';

%% =========================================================
% STORAGE
%% =========================================================
AllRows = table();

fprintf('=========================================================\n');
fprintf('Stage 2 single-model sparse screening\n');
fprintf('Using battery-specific Q_nom_init_per_battery from MAT\n');
fprintf('cycle_stride_train = %d\n', cycle_stride_train);
fprintf('base_step = %d\n', base_step);
fprintf('n_region1 = %d | n_region2 = %d | n_region3 = %d | n_high_innov = %d\n', ...
    n_region1, n_region2, n_region3, n_high_innov);
fprintf('=========================================================\n');

%% =========================================================
% BUILD SAMPLE TABLE
%% =========================================================
for ib = 1:numel(train_batteries)
    battery_no = train_batteries(ib);

    if battery_no > numel(t_all) || isempty(t_all{battery_no}) || ...
       battery_no > numel(I_all) || isempty(I_all{battery_no}) || ...
       battery_no > numel(V_all) || isempty(V_all{battery_no}) || ...
       battery_no > numel(Q_all) || isempty(Q_all{battery_no})
        fprintf('Battery %d skipped: missing data\n', battery_no);
        continue;
    end

    Q_nom_init_batt = Q_nom_init_per_battery(battery_no)*3600;

    if ~isfinite(Q_nom_init_batt) || Q_nom_init_batt <= 0
        fprintf('Battery %d skipped: invalid Q_nom_init_per_battery entry\n', battery_no);
        continue;
    end

    % convert Ah -> As
    Q_nom = Q_nom_init_batt;

    num_cycles = numel(t_all{battery_no});
    cycle_candidates = unique([1, 2:cycle_stride_train:num_cycles, num_cycles]);
    cycle_candidates = cycle_candidates(cycle_candidates >= 1 & cycle_candidates <= num_cycles);

    fprintf('Battery %d | num_cycles = %d | selected cycles = %d | Qnom_init = %.6f Ah\n', ...
        battery_no, num_cycles, numel(cycle_candidates), Q_nom_init_batt);

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
                           numel(debug.dSOC_corr), numel(debug.dOCV_dSOC), ...
                           numel(debug.Rk_eff), numel(debug.soc_gate_alpha)]);

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

            t_sec = (t - t(1)) * 60;
            if max(t_sec) > 0
                tau = t_sec / max(t_sec);
            else
                tau = zeros(size(t_sec));
            end

            abs_innov = abs(innovation);
            innov_abs_ema = causalEMA(abs_innov, 0.05);

            abs_dOCV_dSOC = abs(dOCV_dSOC);
            abs_dOCV_dSOC_ema = causalEMA(abs_dOCV_dSOC, 0.05);

            norm_innov = abs_innov ./ sqrt(max(Rk_eff, 1e-12));
            norm_innov_ema = causalEMA(norm_innov, 0.05);

            abs_I = abs(I_raw);
            throughput_Ah = cumtrapz(t_sec, abs_I) / 3600;

            if Q_nom_cycle_start > 0
                throughput_frac = throughput_Ah * 3600 / Q_nom_cycle_start;
                qnom_start_frac = Q_nom_cycle_start / (Q_nom_init_batt * 3600);
            else
                throughput_frac = zeros(size(throughput_Ah));
                qnom_start_frac = NaN;
            end

            soc_gate_alpha_ema = causalEMA(soc_gate_alpha, 0.05);

            abs_v_resid = abs(V_resid);
            inv_abs_dOCV_dSOC = 1 ./ max(abs_dOCV_dSOC, 1e-8);

            v_resid_abs_mean_so_far = cummean_custom(abs_v_resid);
            inv_abs_dOCV_dSOC_mean_so_far = cummean_custom(inv_abs_dOCV_dSOC);
            norm_innov_mean_so_far = cummean_custom(norm_innov);

            sample_idx = selectSparseIndicesGlobal( ...
                tau, abs_innov, ...
                base_step, n_region1, n_region2, n_region3, n_high_innov, ...
                region1, region2, region3, ...
                use_extra_dense_early_mesh);

            if isempty(sample_idx)
                Q_nom = Q_nom_next;
                continue;
            end

            block = table();
            block.battery_no = repmat(battery_no, numel(sample_idx), 1);
            block.cycle_idx = repmat(cycle_idx, numel(sample_idx), 1);
            block.residual = residual(sample_idx);

            block.soc_est = SOC_est(sample_idx);
            block.tau = tau(sample_idx);
            block.innovation = innovation(sample_idx);
            block.abs_innov = abs_innov(sample_idx);
            block.innov_abs_ema = innov_abs_ema(sample_idx);
            block.norm_innov = norm_innov(sample_idx);
            block.norm_innov_ema = norm_innov_ema(sample_idx);
            block.abs_dOCV_dSOC = abs_dOCV_dSOC(sample_idx);
            block.abs_dOCV_dSOC_ema = abs_dOCV_dSOC_ema(sample_idx);
            block.throughput_frac = throughput_frac(sample_idx);
            block.soc_gate_alpha_ema = soc_gate_alpha_ema(sample_idx);
            block.v_resid_abs_mean_so_far = v_resid_abs_mean_so_far(sample_idx);
            block.inv_abs_dOCV_dSOC_mean_so_far = inv_abs_dOCV_dSOC_mean_so_far(sample_idx);
            block.norm_innov_mean_so_far = norm_innov_mean_so_far(sample_idx);

            if use_qnom_start_frac
                block.qnom_start_frac = repmat(qnom_start_frac, numel(sample_idx), 1);
            end

            AllRows = [AllRows; block]; %#ok<AGROW>

            Q_nom = Q_nom_next;

        catch err
            fprintf('Battery %d | cycle %d error: %s\n', battery_no, cycle_idx, err.message);
            continue;
        end
    end
end

if isempty(AllRows)
    error('No samples collected.');
end

%% =========================================================
% SIMPLE FEATURE SCORING
%% =========================================================
scoreRows = [];

for i = 1:numel(candidateFeatures)
    fname = candidateFeatures{i};

    if ~ismember(fname, AllRows.Properties.VariableNames)
        continue;
    end

    x = AllRows.(fname);
    valid = isfinite(x) & isfinite(AllRows.residual);

    if nnz(valid) < 30
        continue;
    end

    rhoS = corr(x(valid), AllRows.residual(valid), 'Type', 'Spearman', 'Rows', 'complete');
    rhoP = corr(x(valid), AllRows.residual(valid), 'Type', 'Pearson',  'Rows', 'complete');

    scoreRows = [scoreRows; {string(fname), abs(rhoS), abs(rhoP)}]; %#ok<AGROW>
end

ScoreTable = cell2table(scoreRows, ...
    'VariableNames', {'feature_name','abs_spearman','abs_pearson'});

ScoreTable.composite_score = 0.7 * ScoreTable.abs_spearman + 0.3 * ScoreTable.abs_pearson;
ScoreTable = sortrows(ScoreTable, 'composite_score', 'descend');

TopPool = ScoreTable(1:min(10,height(ScoreTable)), :);

writetable(ScoreTable, scores_csv);
writetable(TopPool,    top_csv);

disp(' ');
disp('==================== STAGE 2 FEATURE SCORES ====================');
disp(ScoreTable);

fprintf('\\nFinal sample count used for scoring: %d\\n', height(AllRows));
fprintf('Saved:\n');
fprintf('  %s\n', scores_csv);
fprintf('  %s\n', top_csv);

%% =========================================================
% LOCAL FUNCTIONS
%% =========================================================
function idx = selectSparseIndicesGlobal(tau, abs_innov, ...
    base_step, n_region1, n_region2, n_region3, n_high_innov, ...
    region1, region2, region3, use_extra_dense_early_mesh)

    N = numel(tau);
    idx = [];

    % -------- base grid --------
    idx_base = unique(round(linspace(1, N, max(2, ceil(N / base_step)))));
    idx = [idx; idx_base(:)];

    % -------- region-focused sampling --------
    idx = [idx; sampleRegion(tau, region1, n_region1)];
    idx = [idx; sampleRegion(tau, region2, n_region2)];
    idx = [idx; sampleRegion(tau, region3, n_region3)];

    % -------- high innovation points --------
    valid = isfinite(abs_innov);
    if any(valid)
        [~, order] = sort(abs_innov(valid), 'descend');
        validIdx = find(valid);
        pick = validIdx(order(1:min(n_high_innov, numel(order))));
        idx = [idx; pick(:)];
    end

    % -------- extra dense early-cycle coverage (global) --------
    if use_extra_dense_early_mesh
        idx = [idx; sampleRegion(tau, [0.00 0.05], 16)];
        idx = [idx; sampleRegion(tau, [0.05 0.15], 16)];
        idx = [idx; sampleRegion(tau, [0.15 0.30], 18)];

        idx_dense = unique(round(linspace(1, N, min(max(40, round(N/8)), N))));
        idx = [idx; idx_dense(:)];
    end

    idx = unique(idx);
    idx = idx(idx >= 1 & idx <= N);
end

function idx = sampleRegion(tau, region, n_pick)
    mask = tau >= region(1) & tau <= region(2);
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
    debug.dSOC_corr = zeros(N,1);
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
        debug.dSOC_corr(k) = x(1) - x_pred(1);
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
