%% =========================================================
% Hybrid EKF-GPR PRIMARY TESTING + Q_NOM CLAMP + SHAPE ASSESSMENT
% EKF vs RAW GPR ONLY
%
% GOAL
%   Run PRIMARY testing of the already-trained Hybrid EKF-GPR model
%   and assign each PRIMARY test cycle to one of the previously
%   discovered shape clusters, while:
%     - IGNORING shape 1 as outlier in final shape-wise assessment
%     - IGNORING Lite model completely
%
% WHAT IT DOES
%   1) Runs EKF on all PRIMARY test cycles
%   2) Applies raw GPR correction
%   3) Builds EKF residual profile per cycle
%   4) Standardizes residual profile exactly as in clustering stage
%   5) Projects the test residual profile to the saved SHAPE PCA space
%   6) Assigns the nearest shape centroid
%   7) Summarizes performance by:
%        - cycle
%        - battery
%        - assigned shape cluster
%   8) Excludes shape 1 from shape-performance summary
%
% IMPORTANT
%   - No retraining
%   - No saving of t_all / I_all / V_all / Q_all
%   - Uses residual_shape_model.mat only to recover the SHAPE space
%   - Lite model is NOT evaluated or saved
%
% REQUIRED FILES
%   - battery_workspace_core.mat          (if core vars not already loaded)
%   - fusion_full_model.mat               (if EKF params/noise not already loaded)
%   - Q_nom_init_first_cycle_all_batteries.mat
%   - hybrid_soc_model.mat
%   - residual_shape_model.mat
%
% OUTPUTS
%   - final_test_hybrid_soc_qnomclamp_battery_summary_primary_withshape.csv
%   - final_test_hybrid_soc_qnomclamp_shape_summary_primary.csv
%   - final_test_hybrid_soc_qnomclamp_primary_withshape_outputs.mat
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

if exist('gpr_final_hybrid', 'var') ~= 1 || exist('predictors', 'var') ~= 1
    if exist('hybrid_soc_model.mat','file') == 2
        hybrid_model = load('hybrid_soc_model.mat', 'gpr_final_hybrid', 'predictors', 'model_info');
    gpr_final_hybrid = hybrid_model.gpr_final_hybrid;
    predictors = hybrid_model.predictors;
    model_info = hybrid_model.model_info;
    else
        error('hybrid_soc_model.mat not found.');
    end
end

if exist('residual_shape_model.mat','file') ~= 2
    error('residual_shape_model.mat not found.');
end
ShapeBundle = load('residual_shape_model.mat');

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
elseif isfield(ShapeBundle, 'train_batteries') && ~isempty(ShapeBundle.train_batteries)
    base_train_batteries = ShapeBundle.train_batteries(:);
end

train_batteries_eval = unique([base_train_batteries; added_train_batteries(:)], 'stable')';

% ---------------------------------------------------------
% PRIMARY TEST SPLIT ONLY
% ---------------------------------------------------------
primary_pool = 1:min(84, numel(t_all));
exclude_batteries = unique([train_batteries_eval(:); 15]);
primary_test_batteries = setdiff(primary_pool, exclude_batteries, 'stable');
test_batteries = primary_test_batteries(:)';

min_cycle_length = 30;
residual_clamp = 0.03;

% -------- Q_nom sanity clamp only --------
qnom_ratio_min = 0.60;
qnom_ratio_max = 1.05;
qnom_max_step_per_cycle = 0.03;

% -------- shape handling --------
outlier_shape_id = 1;      % ignore this in shape summary / selected export
keep_only_nonoutlier_selected = true;

% -------- selected-cycle exports for plots --------
save_selected_cycle_tables = true;
nGrid = 201;
tauGrid_plot = linspace(0,1,nGrid)';

late_window = [0.70 0.98];
mid_window  = [0.15 0.60];
phaseNames = {'FIRST','MIDDLE','LAST'};

% -------- filenames --------
save_csv = true;
summary_csv_name  = 'final_test_hybrid_soc_qnomclamp_battery_summary_primary_withshape.csv';
shape_csv_name    = 'final_test_hybrid_soc_qnomclamp_shape_summary_primary.csv';

%% =========================================================
% CHECKS
%% =========================================================
requiredVars = { ...
    't_all','I_all','V_all','Q_all', ...
    'Q_nom_init_per_battery', ...
    'I_noise_std','V_noise_std', ...
    'R0','R1','C1','R2','C2', ...
    'gpr_final_hybrid','predictors'};

for k = 1:numel(requiredVars)
    if exist(requiredVars{k}, 'var') ~= 1
        error('Missing variable: %s', requiredVars{k});
    end
end

Q_nom_init_per_battery = Q_nom_init_per_battery(:);

%% =========================================================
% RECOVER SHAPE MODEL FROM residual_shape_model.mat
%% =========================================================
requiredShapeFields = {'ResidualShape','score_shape','bestCtr_shape','tau_grid'};
for k = 1:numel(requiredShapeFields)
    if ~isfield(ShapeBundle, requiredShapeFields{k})
        error('residual_shape_model.mat is missing %s', requiredShapeFields{k});
    end
end

ResidualShape_train = ShapeBundle.ResidualShape;
score_shape_train   = ShapeBundle.score_shape;
bestCtr_shape       = ShapeBundle.bestCtr_shape;
tau_grid_shape      = ShapeBundle.tau_grid(:);

if isfield(ShapeBundle, 'max_pca_dims_for_clustering')
    nPC_shape = min(ShapeBundle.max_pca_dims_for_clustering, size(score_shape_train,2));
else
    nPC_shape = size(bestCtr_shape,2);
end
nPC_shape = min(nPC_shape, size(bestCtr_shape,2));

[coeff_shape_rec, score_shape_rec, ~, ~, explained_shape_rec, mu_shape_rec] = ...
    pca(ResidualShape_train, 'Rows', 'complete');

coeff_shape_use = coeff_shape_rec(:,1:nPC_shape);
score_shape_use = score_shape_rec(:,1:nPC_shape);

for j = 1:nPC_shape
    a = score_shape_use(:,j);
    b = score_shape_train(:,j);
    valid = isfinite(a) & isfinite(b);
    if any(valid)
        c = corr(a(valid), b(valid), 'Rows', 'complete');
        if isfinite(c) && c < 0
            coeff_shape_use(:,j) = -coeff_shape_use(:,j);
            score_shape_use(:,j) = -score_shape_use(:,j);
        end
    end
end

shape_model = struct();
shape_model.tau_grid = tau_grid_shape;
shape_model.mu_train = mu_shape_rec(:)';
shape_model.coeff = coeff_shape_use;
shape_model.centroids = bestCtr_shape(:,1:nPC_shape);
shape_model.nPC = nPC_shape;
shape_model.explained = explained_shape_rec(1:nPC_shape);

fprintf('\n=========================================================\n');
fprintf('PRIMARY TESTING + SHAPE ASSESSMENT\n');
fprintf('Recovered shape space from residual_shape_model.mat\n');
fprintf('Shape clusters         : %d\n', size(shape_model.centroids,1));
fprintf('Shape PCA dimensions   : %d\n', shape_model.nPC);
fprintf('Primary test batteries : %d\n', numel(primary_test_batteries));
fprintf('Ignoring outlier shape : %d\n', outlier_shape_id);
fprintf('=========================================================\n');

%% =========================================================
% MAIN TEST ON ALL CYCLES OF PRIMARY TEST BATTERIES
%% =========================================================
CycleResults = table();
BatterySummary = table();
ShapeSummary = table();

SelRows = {};
TrajRows = {};

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
                residual_pred(valid_te) = predict(gpr_final_hybrid, Xtest(valid_te,:));
            end
            residual_pred = max(min(residual_pred, residual_clamp), -residual_clamp);

            % ---------------------------------------------------------
            % RAW GPR correction only
            % ---------------------------------------------------------
            SOC_gpr = SOC_est + residual_pred;
            SOC_gpr = min(max(SOC_gpr, 0), 1);

            % ---------------------------------------------------------
            % RMSEs
            % ---------------------------------------------------------
            rmse_ekf  = sqrt(mean((SOC_est - SOC_true).^2, 'omitnan'));
            rmse_gpr  = sqrt(mean((SOC_gpr - SOC_true).^2, 'omitnan'));

            lateMask = progress_causal >= 0.80;

            if any(lateMask)
                rmse_ekf_late  = sqrt(mean((SOC_est(lateMask) - SOC_true(lateMask)).^2, 'omitnan'));
                rmse_gpr_late  = sqrt(mean((SOC_gpr(lateMask) - SOC_true(lateMask)).^2, 'omitnan'));
            else
                rmse_ekf_late  = NaN;
                rmse_gpr_late  = NaN;
            end

            delta_gpr_vs_ekf = rmse_gpr - rmse_ekf;
            delta_gpr_vs_ekf_late = rmse_gpr_late - rmse_ekf_late;

            % ---------------------------------------------------------
            % SHAPE ASSIGNMENT USING EKF RESIDUAL PROFILE
            % ---------------------------------------------------------
            residual_true_ekf = SOC_true - SOC_est;
            tau_cycle = build_tau_plot(t);

            [shape_profile_interp, shape_profile_z] = ...
                build_shape_profile(residual_true_ekf, tau_cycle, shape_model.tau_grid);

            if all(~isfinite(shape_profile_z))
                shape_cluster_assigned = NaN;
                shape_distance_min = NaN;
                shape_score_1 = NaN;
                shape_score_2 = NaN;
                shape_score_3 = NaN;
                shape_score_4 = NaN;
                shape_score_5 = NaN;
            else
                shape_score = project_shape_profile(shape_profile_z, shape_model);
                [shape_cluster_assigned, shape_distance_min] = ...
                    nearest_centroid(shape_score, shape_model.centroids);

                shape_score_pad = nan(1,5);
                shape_score_pad(1:min(numel(shape_score),5)) = shape_score(1:min(numel(shape_score),5));

                shape_score_1 = shape_score_pad(1);
                shape_score_2 = shape_score_pad(2);
                shape_score_3 = shape_score_pad(3);
                shape_score_4 = shape_score_pad(4);
                shape_score_5 = shape_score_pad(5);
            end

            rmse_ekf_all = [rmse_ekf_all; rmse_ekf]; %#ok<AGROW>
            rmse_gpr_all = [rmse_gpr_all; rmse_gpr]; %#ok<AGROW>
            rmse_ekf_late_all = [rmse_ekf_late_all; rmse_ekf_late]; %#ok<AGROW>
            rmse_gpr_late_all = [rmse_gpr_late_all; rmse_gpr_late]; %#ok<AGROW>
            qnom_clamped_all = [qnom_clamped_all; qnom_was_clamped]; %#ok<AGROW>

            is_selected_cycle = double(ismember(cycle_idx, selected_cycles));

            row_count = row_count + 1;
            RowCell{row_count} = table( ...
                battery_no, cycle_idx, ...
                Q_nom_cycle_start / Q_nom_init_batt, ...
                qnom_was_clamped, ...
                rmse_ekf, rmse_gpr, ...
                delta_gpr_vs_ekf, ...
                rmse_ekf_late, rmse_gpr_late, ...
                delta_gpr_vs_ekf_late, ...
                mean(abs(residual_pred), 'omitnan'), ...
                rmse_V, ...
                is_selected_cycle, ...
                shape_cluster_assigned, ...
                shape_distance_min, ...
                shape_score_1, shape_score_2, shape_score_3, shape_score_4, shape_score_5, ...
                'VariableNames', { ...
                'battery_no','cycle_idx','qnom_start_frac', ...
                'qnom_was_clamped', ...
                'rmse_ekf','rmse_gpr', ...
                'delta_gpr_vs_ekf', ...
                'rmse_ekf_late','rmse_gpr_late', ...
                'delta_gpr_vs_ekf_late', ...
                'mean_abs_residual_pred','rmse_v', ...
                'is_selected_cycle', ...
                'shape_cluster', ...
                'shape_distance_min', ...
                'shape_score_1','shape_score_2','shape_score_3','shape_score_4','shape_score_5'});

            % ---------------------------------------------------------
            % SELECTED-CYCLE EXPORT TABLES FOR PLOTS
            % ---------------------------------------------------------
            if save_selected_cycle_tables && is_selected_cycle
                if ~(keep_only_nonoutlier_selected && shape_cluster_assigned == outlier_shape_id)
                    abs_err_ekf = abs(SOC_true - SOC_est);
                    abs_err_gpr = abs(SOC_true - SOC_gpr);

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

                    resid_ekf_i = interp1_monotonic_safe(tau_cycle, SOC_true - SOC_est, tauGrid_plot);
                    resid_gpr_i = interp1_monotonic_safe(tau_cycle, SOC_true - SOC_gpr, tauGrid_plot);
                    abs_err_ekf_i = interp1_monotonic_safe(tau_cycle, abs_err_ekf, tauGrid_plot);
                    abs_err_gpr_i = interp1_monotonic_safe(tau_cycle, abs_err_gpr, tauGrid_plot);

                    soc_true_i = interp1_monotonic_safe(tau_cycle, SOC_true, tauGrid_plot);
                    soc_ekf_i  = interp1_monotonic_safe(tau_cycle, SOC_est,  tauGrid_plot);
                    soc_gpr_i  = interp1_monotonic_safe(tau_cycle, SOC_gpr,  tauGrid_plot);

                    v_true_i   = interp1_monotonic_safe(tau_cycle, V_raw,   tauGrid_plot);
                    v_model_i  = interp1_monotonic_safe(tau_cycle, V_model, tauGrid_plot);

                    SelRows(end+1,1) = {table( ...
                        battery_no, cycle_idx, string(phase_label), ...
                        Q_nom_cycle_start / Q_nom_init_batt, ...
                        qnom_was_clamped, ...
                        rmse_ekf, rmse_gpr, ...
                        delta_gpr_vs_ekf, ...
                        res_energy_ekf, res_energy_gpr, ...
                        late_peak_ekf, late_t_ekf, ...
                        late_peak_gpr, late_t_gpr, ...
                        mid_valley_ekf, mid_t_ekf, ...
                        mid_valley_gpr, mid_t_gpr, ...
                        rmse_V, ...
                        shape_cluster_assigned, ...
                        shape_distance_min, ...
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
                        'rmse_v', ...
                        'shape_cluster', ...
                        'shape_distance_min'})};

                    TrajRows(end+1,1) = {table( ...
                        repmat(battery_no, nGrid, 1), ...
                        repmat(cycle_idx, nGrid, 1), ...
                        repmat(string(phase_label), nGrid, 1), ...
                        repmat(shape_cluster_assigned, nGrid, 1), ...
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
                        'battery_no','cycle_idx','phase','shape_cluster','tau', ...
                        'soc_true','soc_ekf','soc_gpr', ...
                        'v_true','v_model', ...
                        'resid_ekf','resid_gpr', ...
                        'abs_err_ekf','abs_err_gpr'})};
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
            mean(rmse_ekf_late_all,'omitnan'), ...
            mean(rmse_gpr_late_all,'omitnan'), ...
            sum(rmse_gpr_all < rmse_ekf_all,'omitnan'), ...
            sum(rmse_gpr_all > rmse_ekf_all,'omitnan'), ...
            mean(rmse_gpr_all < rmse_ekf_all,'omitnan'), ...
            mean(rmse_gpr_all > rmse_ekf_all,'omitnan'), ...
            sum(qnom_clamped_all,'omitnan'), ...
            'VariableNames', { ...
            'battery_no','num_cycles_evaluated', ...
            'mean_rmse_ekf','mean_rmse_gpr', ...
            'mean_rmse_ekf_late','mean_rmse_gpr_late', ...
            'num_gpr_improved_vs_ekf', ...
            'num_gpr_worse_vs_ekf', ...
            'gpr_improve_rate_vs_ekf', ...
            'gpr_failure_rate_vs_ekf', ...
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

if ~isempty(SelRows)
    GPRSelTable = vertcat(SelRows{:});
    GPRSelTable.phase = categorical(string(GPRSelTable.phase), phaseNames, 'Ordinal', true);
else
    GPRSelTable = table();
end

if ~isempty(TrajRows)
    GPRTrajTable = vertcat(TrajRows{:});
    GPRTrajTable.phase = categorical(string(GPRTrajTable.phase), phaseNames, 'Ordinal', true);
else
    GPRTrajTable = table();
end

%% =========================================================
% SHAPE SUMMARY
% IGNORE OUTLIER SHAPE
%% =========================================================
if ~isempty(CycleResults) && ismember('shape_cluster', CycleResults.Properties.VariableNames)
    valid_shape = isfinite(CycleResults.shape_cluster) & CycleResults.shape_cluster ~= outlier_shape_id;
    Tshape = CycleResults(valid_shape, :);

    if ~isempty(Tshape)
        [G, K] = findgroups(Tshape.shape_cluster);

        ShapeSummary = table();
        ShapeSummary.shape_cluster = K;
        ShapeSummary.n_cycles = splitapply(@numel, Tshape.cycle_idx, G);
        ShapeSummary.n_batteries = splitapply(@(x) numel(unique(x)), Tshape.battery_no, G);

        ShapeSummary.mean_rmse_ekf = splitapply(@(x) mean(x,'omitnan'), Tshape.rmse_ekf, G);
        ShapeSummary.mean_rmse_gpr = splitapply(@(x) mean(x,'omitnan'), Tshape.rmse_gpr, G);

        ShapeSummary.mean_delta_gpr_vs_ekf = splitapply(@(x) mean(x,'omitnan'), Tshape.delta_gpr_vs_ekf, G);

        ShapeSummary.mean_rmse_ekf_late = splitapply(@(x) mean(x,'omitnan'), Tshape.rmse_ekf_late, G);
        ShapeSummary.mean_rmse_gpr_late = splitapply(@(x) mean(x,'omitnan'), Tshape.rmse_gpr_late, G);

        ShapeSummary.gpr_improve_rate_vs_ekf = splitapply(@(x) mean(x < 0,'omitnan'), Tshape.delta_gpr_vs_ekf, G);
        ShapeSummary.gpr_failure_rate_vs_ekf = splitapply(@(x) mean(x > 0,'omitnan'), Tshape.delta_gpr_vs_ekf, G);

        ShapeSummary.mean_qnom_start_frac = splitapply(@(x) mean(x,'omitnan'), Tshape.qnom_start_frac, G);
        ShapeSummary.mean_shape_distance = splitapply(@(x) mean(x,'omitnan'), Tshape.shape_distance_min, G);

        ShapeSummary = sortrows(ShapeSummary, 'shape_cluster');
    end
end

%% =========================================================
% DISPLAY
%% =========================================================
disp(' ');
disp('==================== PRIMARY CYCLE RESULTS (HEAD) ====================');
if ~isempty(CycleResults)
    disp(CycleResults(1:min(20,height(CycleResults)), :));
else
    disp('CycleResults is empty.');
end

disp(' ');
disp('==================== PRIMARY BATTERY SUMMARY ====================');
if ~isempty(BatterySummary)
    disp(BatterySummary);
else
    disp('BatterySummary is empty.');
end

disp(' ');
disp('==================== PRIMARY SHAPE SUMMARY (OUTLIER REMOVED) ====================');
if ~isempty(ShapeSummary)
    disp(ShapeSummary);
else
    disp('ShapeSummary is empty.');
end

if ~isempty(BatterySummary)
    fprintf('\n---------------- PRIMARY TESTING ----------------\n');
    fprintf('Mean EKF RMSE         = %.6f\n', mean(BatterySummary.mean_rmse_ekf,'omitnan'));
    fprintf('Mean GPR RMSE         = %.6f\n', mean(BatterySummary.mean_rmse_gpr,'omitnan'));
    fprintf('Mean EKF late RMSE    = %.6f\n', mean(BatterySummary.mean_rmse_ekf_late,'omitnan'));
    fprintf('Mean GPR late RMSE    = %.6f\n', mean(BatterySummary.mean_rmse_gpr_late,'omitnan'));
    fprintf('Mean GPR improve rate = %.2f %%\n', 100 * mean(BatterySummary.gpr_improve_rate_vs_ekf,'omitnan'));
    fprintf('Mean GPR failure rate = %.2f %%\n', 100 * mean(BatterySummary.gpr_failure_rate_vs_ekf,'omitnan'));
    fprintf('Total Q_nom clamped cycles = %d\n', sum(BatterySummary.num_qnom_clamped_cycles,'omitnan'));
end

if ~isempty(ShapeSummary)
    fprintf('\n---------------- SHAPE-WISE PERFORMANCE (OUTLIER REMOVED) ----------------\n');
    for i = 1:height(ShapeSummary)
        fprintf('Shape %d | n=%d | mean Δ(GPR-EKF)=%.6f | GPR improve rate=%.2f %%\n', ...
            ShapeSummary.shape_cluster(i), ...
            ShapeSummary.n_cycles(i), ...
            ShapeSummary.mean_delta_gpr_vs_ekf(i), ...
            100*ShapeSummary.gpr_improve_rate_vs_ekf(i));
    end
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

    if ~isempty(ShapeSummary)
        writetable(ShapeSummary, shape_csv_name);
    else
        writetable(table(), shape_csv_name);
    end

    fprintf('\nSaved CSV files:\n');    fprintf('  %s\n', summary_csv_name);
    fprintf('  %s\n', shape_csv_name);end

%% =========================================================
% SAVE MAT OUTPUTS
%% =========================================================

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

function [r_interp, r_shape] = build_shape_profile(residual, tau, tau_grid)
    residual = residual(:);
    tau = tau(:);
    tau_grid = tau_grid(:);

    valid = isfinite(residual) & isfinite(tau);
    residual = residual(valid);
    tau = tau(valid);

    r_interp = nan(size(tau_grid));
    r_shape = nan(size(tau_grid));

    if numel(residual) < 5
        return;
    end

    [tau_u, ia] = unique(tau, 'stable');
    if numel(tau_u) < 5
        return;
    end

    resid_u = residual(ia);
    r_interp = interp1(tau_u, resid_u, tau_grid, 'linear', 'extrap');

    mu_r = mean(r_interp, 'omitnan');
    sd_r = std(r_interp, 0, 'omitnan');

    if ~isfinite(sd_r) || sd_r < eps
        r_shape = zeros(size(r_interp));
    else
        r_shape = (r_interp - mu_r) ./ sd_r;
    end
end

function score = project_shape_profile(r_shape, shape_model)
    x = r_shape(:)';
    x_centered = x - shape_model.mu_train;
    score = x_centered * shape_model.coeff;
end

function [kbest, dmin] = nearest_centroid(score, C)
    if any(~isfinite(score))
        kbest = NaN;
        dmin = NaN;
        return;
    end
    d = sqrt(sum((C - score).^2, 2));
    [dmin, kbest] = min(d);
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