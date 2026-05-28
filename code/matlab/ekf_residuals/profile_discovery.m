%% =========================================================
% TRAINING RESIDUAL PROFILE DISCOVERY (PUBLICATION VERSION)
%
% GOAL:
%   Detect whether multiple EKF residual profile families exist
%   in the TRAINING data and generate publication-grade evidence
%   that the residual is structured and not random.
%
% REQUIRED IN WORKSPACE:
%   t_all, I_all, V_all, Q_all
%
% REQUIRED FILE:
%   Q_nom_init_first_cycle_all_batteries.mat
%
% OUTPUTS:
%   residual_profile_cycle_table.csv
%   residual_profile_cluster_summary.csv
%   residual_profile_k_selection.csv
%   residual_profile_publication_summary.csv
%   residual_profile_cyclelabel_summary.csv
%   residual_profile_shape_cluster_label_crosstab.csv
%   residual_profile_raw_cluster_label_crosstab.csv
%
% FIGURES:
%   - PCA scatter (raw residual)
%   - PCA scatter (shape-only residual)
%   - Mean residual profile per cluster
%   - Mean shape-only profile per cluster
%   - K selection metrics
%   - SOH vs cluster
%   - Mean residual by cycle label
%   - Cluster composition by cycle label
%   - Residual heatmap sorted by shape cluster
%% =========================================================

% clear; clc; close all;
rng(42);

%% =========================================================
% SETTINGS
%% =========================================================
train_batteries = [2 4 6 8 9 10 12 14 16 18 19 21 23 25 27 29 31 33 35 37 39 41 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82];

I_noise_std = 0.05;
V_noise_std = 0.03;

R0 = 0.0167;
R1 = 0.01;   C1 = 3000;
R2 = 0.03;   C2 = 2000;

min_cycle_length = 30;

% normalized cycle grid for residual profile
n_tau = 150;
tau_grid = linspace(0,1,n_tau)';

% clustering settings
K_list = 2:6;
n_replicates = 10;
max_pca_dims_for_clustering = 5;

% optional: remove known extreme outlier cycles
remove_known_outliers = true;
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

save_csv = true;

%% =========================================================
% LOAD BATTERY-SPECIFIC Q_NOM_INIT
%% =========================================================
if exist('Q_nom_init_per_battery', 'var') ~= 1
    load('Q_nom_init_first_cycle_all_batteries.mat', 'Q_nom_init_per_battery');
end

Q_nom_init_per_battery = Q_nom_init_per_battery(:);

%% =========================================================
% CHECKS
%% =========================================================
requiredVars = {'t_all','I_all','V_all','Q_all','Q_nom_init_per_battery'};
for k = 1:numel(requiredVars)
    if exist(requiredVars{k}, 'var') ~= 1
        error('Missing variable: %s', requiredVars{k});
    end
end

%% =========================================================
% OCV FUNCTION
%% =========================================================
OCV_func = @(soc) interp1( ...
    [0 0.02 0.05 0.10 0.20 0.40 0.60 0.80 0.90 0.95 0.98 1.00], ...
    [2.00 2.75 3.05 3.18 3.24 3.27 3.29 3.31 3.33 3.35 3.39 3.60], ...
    soc, 'pchip', 'extrap');

%% =========================================================
% STORAGE
%% =========================================================
ResidualRaw = [];
ResidualShape = [];

battery_vec = [];
cycle_vec = [];
num_cycles_vec = [];
cycle_frac_vec = [];
cycle_label_vec = strings(0,1);

qnom_init_ah_vec = [];
soh_true_vec = [];
qnom_start_vec = [];

rmse_ekf_vec = [];
rmse_v_vec = [];

residual_mean_vec = [];
residual_std_vec = [];
residual_late_mean_vec = [];
residual_end_minus_mid_vec = [];
residual_abs_area_vec = [];
residual_peak_abs_vec = [];

fprintf('=========================================================\n');
fprintf('Building training residual profile dataset\n');
fprintf('Training batteries: %s\n', mat2str(train_batteries));
fprintf('Using battery-specific Q_nom_init from MAT file\n');
fprintf('=========================================================\n');
SOCTrueProfiles = [];
SOCEstProfiles  = [];
VoltageTrueProfiles  = [];
VoltageModelProfiles = [];
%% =========================================================
% MAIN LOOP
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

    if battery_no > numel(Q_nom_init_per_battery) || ...
       ~isfinite(Q_nom_init_per_battery(battery_no)) || ...
       Q_nom_init_per_battery(battery_no) <= 0
        fprintf('Battery %d skipped: invalid Q_nom_init_per_battery\n', battery_no);
        continue;
    end

    num_cycles = numel(t_all{battery_no});
    if num_cycles < 1
        continue;
    end

    fprintf('Battery %d | num_cycles = %d\n', battery_no, num_cycles);

    Q_nom_init_batt_Ah = Q_nom_init_per_battery(battery_no);
    Q_nom_init_batt_C  = Q_nom_init_batt_Ah * 3600;

    Q_nom = Q_nom_init_batt_C;

    for cycle_idx = 1:num_cycles
        try
            if remove_known_outliers
                if any(bad_training_cycles(:,1) == battery_no & bad_training_cycles(:,2) == cycle_idx)
                    continue;
                end
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

            [SOC_est, V_model, rmse_V, Q_accumulated, debug] = ...
                ekf_thevenin_2RC_R0_adaptive_v2( ...
                    t_data, I_data_noisy, V_data_noisy, ...
                    Q_nom, R0, R1, C1, R2, C2, OCV_func);

            if isfinite(Q_accumulated/2) && (Q_accumulated/2) > 0
                Q_nom = Q_accumulated / 2;
            end

            qmax = max(Q_series);
            if ~isfinite(qmax) || qmax <= 0
                continue;
            end

            min_len = min([numel(t_data), numel(Q_series), numel(SOC_est), numel(V_model)]);
            t = t_data(1:min_len);
            Qs = Q_series(1:min_len);
            SOC_est = SOC_est(1:min_len);
            V_model = V_model(1:min_len);
            V_used = V_data(1:min_len);

            SOC_true = Qs / qmax;
            SOC_true = min(max(SOC_true, 0), 1);

            residual = SOC_true - SOC_est;
            residual(~isfinite(residual)) = NaN;

            if all(~isfinite(residual))
                continue;
            end

            if t(end) > t(1)
                tau = (t - t(1)) / (t(end) - t(1));
            else
                tau = linspace(0,1,numel(t))';
            end

            [tau_u, ia] = unique(tau, 'stable');
            if numel(tau_u) < 5
                continue;
            end
            residual_u = residual(ia);
 r_interp = interp1(tau_u, residual_u, tau_grid, 'linear', 'extrap');
            % raw residual profile
            soc_true_u = SOC_true(ia);
soc_est_u  = SOC_est(ia);

soc_true_interp = interp1(tau_u, soc_true_u, tau_grid, 'linear', 'extrap');
soc_est_interp  = interp1(tau_u, soc_est_u,  tau_grid, 'linear', 'extrap');

SOCTrueProfiles = [SOCTrueProfiles; soc_true_interp']; %#ok<AGROW>
SOCEstProfiles  = [SOCEstProfiles;  soc_est_interp'];  %#ok<AGROW>
V_true_u  = V_used(ia);
V_model_u = V_model(ia);

v_true_interp  = interp1(tau_u, V_true_u,  tau_grid, 'linear', 'extrap');
v_model_interp = interp1(tau_u, V_model_u, tau_grid, 'linear', 'extrap');

VoltageTrueProfiles  = [VoltageTrueProfiles;  v_true_interp'];   %#ok<AGROW>
VoltageModelProfiles = [VoltageModelProfiles; v_model_interp'];  %#ok<AGROW>            
           

            % shape-only residual profile
            r_shape = r_interp;
            mu_r = mean(r_shape, 'omitnan');
            sd_r = std(r_shape, 0, 'omitnan');
            if ~isfinite(sd_r) || sd_r < eps
                r_shape = zeros(size(r_shape));
            else
                r_shape = (r_shape - mu_r) ./ sd_r;
            end

            % cycle label
            if cycle_idx == 1
                cycle_label = "first";
            elseif cycle_idx == num_cycles
                cycle_label = "last";
            elseif abs(cycle_idx - round(num_cycles/2)) <= max(1, round(0.05*num_cycles))
                cycle_label = "middle";
            else
                cycle_label = "other";
            end

            % extra summaries
            idx_mid = tau_grid > 0.40 & tau_grid < 0.60;
            idx_end = tau_grid >= 0.90;
            idx_late = tau_grid >= 0.80;

            soh_true = qmax / Q_nom_init_batt_Ah;
            rmse_ekf = sqrt(mean((SOC_est - SOC_true).^2, 'omitnan'));
            rmse_v = sqrt(mean((V_used - V_model).^2, 'omitnan'));

            ResidualRaw = [ResidualRaw; r_interp']; %#ok<AGROW>
            ResidualShape = [ResidualShape; r_shape']; %#ok<AGROW>

            battery_vec = [battery_vec; battery_no]; %#ok<AGROW>
            cycle_vec = [cycle_vec; cycle_idx]; %#ok<AGROW>
            num_cycles_vec = [num_cycles_vec; num_cycles]; %#ok<AGROW>
            cycle_frac_vec = [cycle_frac_vec; cycle_idx / num_cycles]; %#ok<AGROW>
            cycle_label_vec = [cycle_label_vec; cycle_label]; %#ok<AGROW>

            qnom_init_ah_vec = [qnom_init_ah_vec; Q_nom_init_batt_Ah]; %#ok<AGROW>
            soh_true_vec = [soh_true_vec; soh_true]; %#ok<AGROW>
            qnom_start_vec = [qnom_start_vec; Q_nom_cycle_start / Q_nom_init_batt_C]; %#ok<AGROW>

            rmse_ekf_vec = [rmse_ekf_vec; rmse_ekf]; %#ok<AGROW>
            rmse_v_vec = [rmse_v_vec; rmse_v]; %#ok<AGROW>

            residual_mean_vec = [residual_mean_vec; mean(r_interp, 'omitnan')]; %#ok<AGROW>
            residual_std_vec = [residual_std_vec; std(r_interp, 0, 'omitnan')]; %#ok<AGROW>
            residual_late_mean_vec = [residual_late_mean_vec; mean(r_interp(idx_late), 'omitnan')]; %#ok<AGROW>
            residual_end_minus_mid_vec = [residual_end_minus_mid_vec; ...
                mean(r_interp(idx_end), 'omitnan') - mean(r_interp(idx_mid), 'omitnan')]; %#ok<AGROW>
            residual_abs_area_vec = [residual_abs_area_vec; trapz(tau_grid, abs(r_interp))]; %#ok<AGROW>
            residual_peak_abs_vec = [residual_peak_abs_vec; max(abs(r_interp), [], 'omitnan')]; %#ok<AGROW>

        catch err
            fprintf('Battery %d | cycle %d error: %s\n', battery_no, cycle_idx, err.message);
            continue;
        end
    end
end

%% =========================================================
% CHECK
%% =========================================================
if isempty(ResidualRaw)
    error('No residual profiles collected.');
end

fprintf('\nCollected %d residual cycles.\n', size(ResidualRaw,1));

%% =========================================================
% BUILD CYCLE TABLE
%% =========================================================
CycleTable = table( ...
    battery_vec, cycle_vec, num_cycles_vec, cycle_frac_vec, cycle_label_vec, ...
    qnom_init_ah_vec, soh_true_vec, qnom_start_vec, rmse_ekf_vec, rmse_v_vec, ...
    residual_mean_vec, residual_std_vec, residual_late_mean_vec, residual_end_minus_mid_vec, ...
    residual_abs_area_vec, residual_peak_abs_vec, ...
    'VariableNames', {'battery_no','cycle_idx','num_cycles','cycle_frac','cycle_label', ...
    'qnom_init_ah','soh_true','qnom_start_frac','rmse_ekf','rmse_v', ...
    'residual_mean','residual_std','residual_late_mean','residual_end_minus_mid', ...
    'residual_abs_area','residual_peak_abs'});

%% =========================================================
% PCA - RAW RESIDUAL
%% =========================================================
[coeff_raw, score_raw, latent_raw, ~, explained_raw] = pca(ResidualRaw, 'Rows', 'complete');
nPC_raw = min(max_pca_dims_for_clustering, size(score_raw,2));
ScoreRawUse = score_raw(:,1:nPC_raw);

%% =========================================================
% PCA - SHAPE-ONLY RESIDUAL
%% =========================================================
[coeff_shape, score_shape, latent_shape, ~, explained_shape] = pca(ResidualShape, 'Rows', 'complete');
nPC_shape = min(max_pca_dims_for_clustering, size(score_shape,2));
ScoreShapeUse = score_shape(:,1:nPC_shape);

%% =========================================================
% K SELECTION
%% =========================================================
KSelRows = [];

bestK_raw = NaN;
bestSil_raw = -inf;
bestIdx_raw = [];
bestCtr_raw = [];

bestK_shape = NaN;
bestSil_shape = -inf;
bestIdx_shape = [];
bestCtr_shape = [];

fprintf('\n=========================================================\n');
fprintf('K-SELECTION FOR RESIDUAL PROFILE CLUSTERING\n');
fprintf('=========================================================\n');

for K = K_list

    % -------- RAW --------
    [idx_raw, ctr_raw] = kmeans(ScoreRawUse, K, ...
        'Replicates', n_replicates, 'Display', 'off');

    sil_raw = mean(silhouette(ScoreRawUse, idx_raw));
    evaCH_raw = evalclusters(ScoreRawUse, idx_raw, 'CalinskiHarabasz');
    evaDB_raw = evalclusters(ScoreRawUse, idx_raw, 'DaviesBouldin');

    % -------- SHAPE --------
    [idx_shape, ctr_shape] = kmeans(ScoreShapeUse, K, ...
        'Replicates', n_replicates, 'Display', 'off');

    sil_shape = mean(silhouette(ScoreShapeUse, idx_shape));
    evaCH_shape = evalclusters(ScoreShapeUse, idx_shape, 'CalinskiHarabasz');
    evaDB_shape = evalclusters(ScoreShapeUse, idx_shape, 'DaviesBouldin');

    KSelRows = [KSelRows; { ...
        K, sil_raw, evaCH_raw.CriterionValues, evaDB_raw.CriterionValues, ...
        sil_shape, evaCH_shape.CriterionValues, evaDB_shape.CriterionValues}]; %#ok<AGROW>

    fprintf('K = %d | RAW sil = %.4f | SHAPE sil = %.4f\n', K, sil_raw, sil_shape);

    if sil_raw > bestSil_raw
        bestSil_raw = sil_raw;
        bestK_raw = K;
        bestIdx_raw = idx_raw;
        bestCtr_raw = ctr_raw;
    end

    if sil_shape > bestSil_shape
        bestSil_shape = sil_shape;
        bestK_shape = K;
        bestIdx_shape = idx_shape;
        bestCtr_shape = ctr_shape;
    end
end

KSelectionTable = cell2table(KSelRows, 'VariableNames', ...
    {'K','silhouette_raw','CH_raw','DB_raw','silhouette_shape','CH_shape','DB_shape'});

fprintf('\nBest K (raw)   = %d | silhouette = %.4f\n', bestK_raw, bestSil_raw);
fprintf('Best K (shape) = %d | silhouette = %.4f\n', bestK_shape, bestSil_shape);

%% =========================================================
% ATTACH CLUSTERS TO CYCLE TABLE
%% =========================================================
CycleTable.cluster_raw = bestIdx_raw;
CycleTable.cluster_shape = bestIdx_shape;

%% =========================================================
% CLUSTER SUMMARY - RAW
%% =========================================================
[Gr, cl_raw] = findgroups(CycleTable.cluster_raw);

ClusterSummaryRaw = table();
ClusterSummaryRaw.cluster_id = cl_raw;
ClusterSummaryRaw.n_cycles = splitapply(@numel, CycleTable.cycle_idx, Gr);
ClusterSummaryRaw.mean_soh = splitapply(@(x) mean(x,'omitnan'), CycleTable.soh_true, Gr);
ClusterSummaryRaw.mean_cycle_frac = splitapply(@(x) mean(x,'omitnan'), CycleTable.cycle_frac, Gr);
ClusterSummaryRaw.mean_rmse_ekf = splitapply(@(x) mean(x,'omitnan'), CycleTable.rmse_ekf, Gr);
ClusterSummaryRaw.mean_rmse_v = splitapply(@(x) mean(x,'omitnan'), CycleTable.rmse_v, Gr);
ClusterSummaryRaw.mean_residual_std = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_std, Gr);
ClusterSummaryRaw.mean_residual_late = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_late_mean, Gr);
ClusterSummaryRaw.mean_end_minus_mid = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_end_minus_mid, Gr);
ClusterSummaryRaw.mean_abs_area = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_abs_area, Gr);
ClusterSummaryRaw.mean_peak_abs = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_peak_abs, Gr);

ClusterSummaryRaw.frac_first = splitapply(@(x) mean(double(x=="first")), CycleTable.cycle_label, Gr);
ClusterSummaryRaw.frac_middle = splitapply(@(x) mean(double(x=="middle")), CycleTable.cycle_label, Gr);
ClusterSummaryRaw.frac_last = splitapply(@(x) mean(double(x=="last")), CycleTable.cycle_label, Gr);
ClusterSummaryRaw.frac_other = splitapply(@(x) mean(double(x=="other")), CycleTable.cycle_label, Gr);

%% =========================================================
% CLUSTER SUMMARY - SHAPE
%% =========================================================
[Gs, cl_shape] = findgroups(CycleTable.cluster_shape);

ClusterSummaryShape = table();
ClusterSummaryShape.cluster_id = cl_shape;
ClusterSummaryShape.n_cycles = splitapply(@numel, CycleTable.cycle_idx, Gs);
ClusterSummaryShape.mean_soh = splitapply(@(x) mean(x,'omitnan'), CycleTable.soh_true, Gs);
ClusterSummaryShape.mean_cycle_frac = splitapply(@(x) mean(x,'omitnan'), CycleTable.cycle_frac, Gs);
ClusterSummaryShape.mean_rmse_ekf = splitapply(@(x) mean(x,'omitnan'), CycleTable.rmse_ekf, Gs);
ClusterSummaryShape.mean_rmse_v = splitapply(@(x) mean(x,'omitnan'), CycleTable.rmse_v, Gs);
ClusterSummaryShape.mean_residual_std = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_std, Gs);
ClusterSummaryShape.mean_residual_late = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_late_mean, Gs);
ClusterSummaryShape.mean_end_minus_mid = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_end_minus_mid, Gs);
ClusterSummaryShape.mean_abs_area = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_abs_area, Gs);
ClusterSummaryShape.mean_peak_abs = splitapply(@(x) mean(x,'omitnan'), CycleTable.residual_peak_abs, Gs);

ClusterSummaryShape.frac_first = splitapply(@(x) mean(double(x=="first")), CycleTable.cycle_label, Gs);
ClusterSummaryShape.frac_middle = splitapply(@(x) mean(double(x=="middle")), CycleTable.cycle_label, Gs);
ClusterSummaryShape.frac_last = splitapply(@(x) mean(double(x=="last")), CycleTable.cycle_label, Gs);
ClusterSummaryShape.frac_other = splitapply(@(x) mean(double(x=="other")), CycleTable.cycle_label, Gs);

%% =========================================================
% MEAN PROFILES PER CLUSTER
%% =========================================================
MeanProfilesRaw = nan(bestK_raw, n_tau);
StdProfilesRaw = nan(bestK_raw, n_tau);

for k = 1:bestK_raw
    Xk = ResidualRaw(CycleTable.cluster_raw == k, :);
    MeanProfilesRaw(k,:) = mean(Xk, 1, 'omitnan');
    StdProfilesRaw(k,:) = std(Xk, 0, 1, 'omitnan');
end

MeanProfilesShape = nan(bestK_shape, n_tau);
StdProfilesShape = nan(bestK_shape, n_tau);

for k = 1:bestK_shape
    Xk = ResidualShape(CycleTable.cluster_shape == k, :);
    MeanProfilesShape(k,:) = mean(Xk, 1, 'omitnan');
    StdProfilesShape(k,:) = std(Xk, 0, 1, 'omitnan');
end
%%

%% =========================================================
% CYCLE LABEL SUMMARY
%% =========================================================
LabelOrder = ["first"; "middle"; "last"; "other"];
LabelSummary = table();

for i = 1:numel(LabelOrder)
    lab = LabelOrder(i);
    idx = CycleTable.cycle_label == lab;

    LabelSummary = [LabelSummary; table( ...
        lab, ...
        sum(idx), ...
        mean(CycleTable.soh_true(idx), 'omitnan'), ...
        mean(CycleTable.cycle_frac(idx), 'omitnan'), ...
        mean(CycleTable.rmse_ekf(idx), 'omitnan'), ...
        mean(CycleTable.rmse_v(idx), 'omitnan'), ...
        mean(CycleTable.residual_mean(idx), 'omitnan'), ...
        mean(CycleTable.residual_std(idx), 'omitnan'), ...
        mean(CycleTable.residual_late_mean(idx), 'omitnan'), ...
        mean(CycleTable.residual_end_minus_mid(idx), 'omitnan'), ...
        mean(CycleTable.residual_abs_area(idx), 'omitnan'), ...
        mean(CycleTable.residual_peak_abs(idx), 'omitnan'), ...
        'VariableNames', {'cycle_label','n_cycles','mean_soh','mean_cycle_frac', ...
        'mean_rmse_ekf','mean_rmse_v','mean_residual_mean','mean_residual_std', ...
        'mean_residual_late_mean','mean_residual_end_minus_mid', ...
        'mean_residual_abs_area','mean_residual_peak_abs'})]; %#ok<AGROW>
end

disp(LabelSummary)
writetable(LabelSummary, 'residual_profile_cyclelabel_summary.csv');
%%

%% =========================================================
% PUBLICATION STATISTICS
%% =========================================================
% ---------------- RAW ----------------
p_kw_soh_raw = kruskalwallis(CycleTable.soh_true, CycleTable.cluster_raw, 'off');
p_kw_cyclefrac_raw = kruskalwallis(CycleTable.cycle_frac, CycleTable.cluster_raw, 'off');
p_kw_rmse_raw = kruskalwallis(CycleTable.rmse_ekf, CycleTable.cluster_raw, 'off');

[ct_raw, chi2_raw, p_chi_raw] = crosstab(CycleTable.cycle_label, categorical(CycleTable.cluster_raw));
n_ct_raw = sum(ct_raw(:));
r_raw = size(ct_raw,1);
c_raw = size(ct_raw,2);
cramersV_raw = sqrt(chi2_raw / max(n_ct_raw * min(r_raw-1, c_raw-1), eps));

within_dist_raw = mean(distance_to_centroid(ScoreRawUse, bestIdx_raw, bestCtr_raw), 'omitnan');
if size(bestCtr_raw,1) > 1
    between_ctr_raw = mean(pdist(bestCtr_raw), 'omitnan');
else
    between_ctr_raw = NaN;
end
sep_ratio_raw = between_ctr_raw / max(within_dist_raw, eps);

% ---------------- SHAPE ----------------
p_kw_soh_shape = kruskalwallis(CycleTable.soh_true, CycleTable.cluster_shape, 'off');
p_kw_cyclefrac_shape = kruskalwallis(CycleTable.cycle_frac, CycleTable.cluster_shape, 'off');
p_kw_rmse_shape = kruskalwallis(CycleTable.rmse_ekf, CycleTable.cluster_shape, 'off');

[ct_shape, chi2_shape, p_chi_shape] = crosstab(CycleTable.cycle_label, categorical(CycleTable.cluster_shape));
n_ct_shape = sum(ct_shape(:));
r_shape = size(ct_shape,1);
c_shape = size(ct_shape,2);
cramersV_shape = sqrt(chi2_shape / max(n_ct_shape * min(r_shape-1, c_shape-1), eps));

within_dist_shape = mean(distance_to_centroid(ScoreShapeUse, bestIdx_shape, bestCtr_shape), 'omitnan');
if size(bestCtr_shape,1) > 1
    between_ctr_shape = mean(pdist(bestCtr_shape), 'omitnan');
else
    between_ctr_shape = NaN;
end
sep_ratio_shape = between_ctr_shape / max(within_dist_shape, eps);

PublicationSummary = table( ...
    ["raw"; "shape"], ...
    [bestK_raw; bestK_shape], ...
    [bestSil_raw; bestSil_shape], ...
    [p_kw_soh_raw; p_kw_soh_shape], ...
    [p_kw_cyclefrac_raw; p_kw_cyclefrac_shape], ...
    [p_kw_rmse_raw; p_kw_rmse_shape], ...
    [p_chi_raw; p_chi_shape], ...
    [cramersV_raw; cramersV_shape], ...
    [within_dist_raw; within_dist_shape], ...
    [between_ctr_raw; between_ctr_shape], ...
    [sep_ratio_raw; sep_ratio_shape], ...
    'VariableNames', {'model_type','bestK','best_silhouette', ...
    'p_kw_soh','p_kw_cycle_frac','p_kw_rmse_ekf', ...
    'p_cyclelabel_cluster_assoc','cramersV_cyclelabel_cluster_assoc', ...
    'mean_within_cluster_distance','mean_between_centroid_distance','separation_ratio'});

%% =========================================================
% DISPLAY
%% =========================================================
disp(' ');
disp('==================== K SELECTION ====================');
disp(KSelectionTable);

disp(' ');
disp('==================== RAW CLUSTER SUMMARY ====================');
disp(ClusterSummaryRaw);

disp(' ');
disp('==================== SHAPE-ONLY CLUSTER SUMMARY ====================');
disp(ClusterSummaryShape);

disp(' ');
disp('==================== CYCLE LABEL SUMMARY ====================');
disp(LabelSummary);

disp(' ');
disp('==================== PUBLICATION SUMMARY ====================');
disp(PublicationSummary);

%% =========================================================
% FIGURE 1: PCA RAW
%% =========================================================
figure('Color','w','Name','PCA Raw Residual Profiles');
gscatter(score_raw(:,1), score_raw(:,2), CycleTable.cluster_raw);
xlabel(sprintf('PC1 (%.1f%%)', explained_raw(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained_raw(2)));
title(sprintf('Raw residual profiles | Best K = %d', bestK_raw));
grid on;

%% =========================================================
% FIGURE 2: PCA SHAPE
%% =========================================================
figure('Color','w','Name','PCA Shape-Only Residual Profiles');
gscatter(score_shape(:,1), score_shape(:,2), CycleTable.cluster_shape);
xlabel(sprintf('PC1 (%.1f%%)', explained_shape(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained_shape(2)));
title(sprintf('Shape-only residual profiles | Best K = %d', bestK_shape));
grid on;

%% =========================================================
% FIGURE 3: RAW CLUSTER MEAN PROFILES
%% =========================================================
figure('Color','w','Name','Raw Residual Mean Profiles');
tiledlayout(bestK_raw,1,'Padding','compact','TileSpacing','compact');

for k = 1:bestK_raw
    nexttile;
    mu = MeanProfilesRaw(k,:)';
    sd = StdProfilesRaw(k,:)';
    fill([tau_grid; flipud(tau_grid)], [mu-sd; flipud(mu+sd)], ...
        [0.85 0.90 1.00], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    hold on;
    plot(tau_grid, mu, 'b-', 'LineWidth', 2);
    yline(0,'k:');
    xlabel('\tau');
    ylabel('Residual');
    title(sprintf('RAW Cluster %d | n = %d', k, sum(CycleTable.cluster_raw==k)));
    grid on;
end

%% =========================================================
% FIGURE 4: SHAPE-ONLY CLUSTER MEAN PROFILES
%% =========================================================
figure('Color','w','Name','Shape-Only Residual Mean Profiles');
tiledlayout(bestK_shape,1,'Padding','compact','TileSpacing','compact');

for k = 1:bestK_shape
    nexttile;
    mu = MeanProfilesShape(k,:)';
    sd = StdProfilesShape(k,:)';
    fill([tau_grid; flipud(tau_grid)], [mu-sd; flipud(mu+sd)], ...
        [1.00 0.88 0.88], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    hold on;
    plot(tau_grid, mu, 'r-', 'LineWidth', 2);
    yline(0,'k:');
    xlabel('\tau');
    ylabel('Standardized residual');
    title(sprintf('SHAPE Cluster %d | n = %d', k, sum(CycleTable.cluster_shape==k)));
    grid on;
end

%% =========================================================
% FIGURE 5: K SELECTION METRICS
%% =========================================================
figure('Color','w','Name','K Selection Metrics');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile;
plot(KSelectionTable.K, KSelectionTable.silhouette_raw, 'bo-', 'LineWidth', 1.5, 'DisplayName','Raw'); hold on;
plot(KSelectionTable.K, KSelectionTable.silhouette_shape, 'rs-', 'LineWidth', 1.5, 'DisplayName','Shape');
xlabel('K'); ylabel('Mean silhouette');
title('Silhouette');
legend('Location','best');
grid on;

nexttile;
plot(KSelectionTable.K, KSelectionTable.CH_raw, 'bo-', 'LineWidth', 1.5, 'DisplayName','Raw'); hold on;
plot(KSelectionTable.K, KSelectionTable.CH_shape, 'rs-', 'LineWidth', 1.5, 'DisplayName','Shape');
xlabel('K'); ylabel('Calinski-Harabasz');
title('CH Index');
legend('Location','best');
grid on;

nexttile;
plot(KSelectionTable.K, KSelectionTable.DB_raw, 'bo-', 'LineWidth', 1.5, 'DisplayName','Raw'); hold on;
plot(KSelectionTable.K, KSelectionTable.DB_shape, 'rs-', 'LineWidth', 1.5, 'DisplayName','Shape');
xlabel('K'); ylabel('Davies-Bouldin');
title('DB Index (lower better)');
legend('Location','best');
grid on;

%% =========================================================
% FIGURE 6: SOH vs cluster
%% =========================================================
figure('Color','w','Name','SOH vs Cluster');
subplot(1,2,1);
boxchart(categorical(CycleTable.cluster_raw), CycleTable.soh_true);
xlabel('Raw cluster');
ylabel('SOH');
title('SOH distribution by raw cluster');
grid on;

subplot(1,2,2);
boxchart(categorical(CycleTable.cluster_shape), CycleTable.soh_true);
xlabel('Shape cluster');
ylabel('SOH');
title('SOH distribution by shape cluster');
grid on;

%% =========================================================
% FIGURE 7: MEAN RESIDUAL BY CYCLE LABEL
%% =========================================================
figure('Color','w','Name','Mean Residual By Cycle Label');
hold on;

label_colors = lines(numel(LabelOrder));

for i = 1:numel(LabelOrder)
    lab = LabelOrder(i);
    idx = CycleTable.cycle_label == lab;
    if ~any(idx), continue; end

    Xlab = ResidualRaw(idx, :);
    mu = mean(Xlab, 1, 'omitnan')';
    sd = std(Xlab, 0, 1, 'omitnan')';

    fill([tau_grid; flipud(tau_grid)], [mu-sd; flipud(mu+sd)], ...
        label_colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.10);
    plot(tau_grid, mu, 'LineWidth', 2, 'Color', label_colors(i,:), 'DisplayName', char(lab));
end

yline(0,'k:');
xlabel('\tau');
ylabel('Residual');
title('Mean EKF residual profile by cycle label');
legend('Location','best');
grid on;

%% =========================================================
% FIGURE 8: SHAPE CLUSTER COMPOSITION BY CYCLE LABEL
%% =========================================================
figure('Color','w','Name','Shape Cluster Composition By Cycle Label');

shape_comp = zeros(bestK_shape, numel(LabelOrder));
for k = 1:bestK_shape
    idxk = CycleTable.cluster_shape == k;
    nk = sum(idxk);
    for i = 1:numel(LabelOrder)
        shape_comp(k,i) = sum(idxk & CycleTable.cycle_label == LabelOrder(i)) / max(nk,1);
    end
end

bar(shape_comp, 'stacked');
xlabel('Shape cluster');
ylabel('Fraction within cluster');
title('Cycle-label composition of shape clusters');
legend(cellstr(LabelOrder), 'Location', 'best');
grid on;

%% =========================================================
% FIGURE 9: HEATMAP SORTED BY SHAPE CLUSTER
%% =========================================================
[~, sort_idx] = sortrows([CycleTable.cluster_shape, CycleTable.cycle_frac, CycleTable.soh_true], [1 2 -3]);

figure('Color','w','Name','Residual Heatmap Sorted By Shape Cluster');
imagesc(tau_grid, 1:size(ResidualRaw,1), ResidualRaw(sort_idx,:));
xlabel('\tau');
ylabel('Cycles (sorted)');
title('Raw residual profiles sorted by shape cluster');
colorbar;
axis tight;

%% =========================================================
% SAVE CSV
%% =========================================================
if save_csv
    writetable(CycleTable, 'residual_profile_cycle_table.csv');

    ClusterSummaryRaw.model_type = repmat("raw", height(ClusterSummaryRaw), 1);
    ClusterSummaryShape.model_type = repmat("shape", height(ClusterSummaryShape), 1);

    ClusterSummaryAll = [ ...
        movevars(ClusterSummaryRaw, 'model_type', 'Before', 1); ...
        movevars(ClusterSummaryShape, 'model_type', 'Before', 1)];

    writetable(ClusterSummaryAll, 'residual_profile_cluster_summary.csv');
    writetable(KSelectionTable, 'residual_profile_k_selection.csv');
    writetable(PublicationSummary, 'residual_profile_publication_summary.csv');
    writetable(LabelSummary, 'residual_profile_cyclelabel_summary.csv');

    RawClusterLabelCrossTab = array2table(ct_raw);
    RawClusterLabelCrossTab.Properties.VariableNames = matlab.lang.makeValidName("cluster_" + string(1:size(ct_raw,2)));
    RawClusterLabelCrossTab.cycle_label = string(categories(categorical(CycleTable.cycle_label)));
    RawClusterLabelCrossTab = movevars(RawClusterLabelCrossTab, 'cycle_label', 'Before', 1);
    writetable(RawClusterLabelCrossTab, 'residual_profile_raw_cluster_label_crosstab.csv');

    ShapeClusterLabelCrossTab = array2table(ct_shape);
    ShapeClusterLabelCrossTab.Properties.VariableNames = matlab.lang.makeValidName("cluster_" + string(1:size(ct_shape,2)));
    ShapeClusterLabelCrossTab.cycle_label = string(categories(categorical(CycleTable.cycle_label)));
    ShapeClusterLabelCrossTab = movevars(ShapeClusterLabelCrossTab, 'cycle_label', 'Before', 1);
    writetable(ShapeClusterLabelCrossTab, 'residual_profile_shape_cluster_label_crosstab.csv');

    fprintf('\nSaved:\n');
    fprintf('  residual_profile_cycle_table.csv\n');
    fprintf('  residual_profile_cluster_summary.csv\n');
    fprintf('  residual_profile_k_selection.csv\n');
    fprintf('  residual_profile_publication_summary.csv\n');
    fprintf('  residual_profile_cyclelabel_summary.csv\n');
    fprintf('  residual_profile_raw_cluster_label_crosstab.csv\n');
    fprintf('  residual_profile_shape_cluster_label_crosstab.csv\n');
end

%% =========================================================
% SAVE SHAPE MODEL BUNDLE FOR PRIMARY TRANSFER ANALYSIS
%% =========================================================
save('residual_plot_bundle.mat', ...
    'ResidualRaw', ...
    'ResidualShape', ...
    'score_shape', ...
    'bestCtr_shape', ...
    'tau_grid', ...
    'max_pca_dims_for_clustering', ...
    'train_batteries', ...
    'CycleTable', ...
    'ClusterSummaryShape', ...
    'ClusterSummaryRaw');

fprintf('  residual_plot_bundle.mat\n');

%% =========================================================
% LOCAL FUNCTIONS
%% =========================================================

function [SOC_est, V_model, rmse_V, Q_accumulated, debug] = ...
    ekf_thevenin_2RC_R0_adaptive_v2(t, I, V_meas, Q_nom, R0, R1, C1, R2, C2, OCV_func)

    N = length(t);
    Q_accumulated = 0;

    x = [0; 0; 0];
    P = diag([1e-4, 1e-4, 1e-4]);

    q_soc  = 1e-10;
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

    debug.t = t(:);
    debug.SOC_cc = zeros(N,1);
    debug.SOC_corr = zeros(N,1);
    debug.dSOC_corr = zeros(N,1);
    debug.innovation = zeros(N,1);
    debug.K_soc = zeros(N,1);
    debug.K_vrc1 = zeros(N,1);
    debug.K_vrc2 = zeros(N,1);
    debug.dOCV_dSOC = zeros(N,1);
    debug.OCV_cc = zeros(N,1);
    debug.OCV_corr = zeros(N,1);
    debug.Rk_eff = zeros(N,1);
    debug.soc_gate_alpha = zeros(N,1);
    debug.slope_factor = zeros(N,1);

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
            slope_factor = slope_boost_factor;
            Rk_eff = min(Rk_eff * slope_factor, R_max);
        else
            slope_factor = 1.0;
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
        debug.SOC_corr(k) = x(1);
        debug.dSOC_corr(k) = x(1) - x_pred(1);
        debug.innovation(k) = y;
        debug.K_soc(k) = K(1);
        debug.K_vrc1(k) = K(2);
        debug.K_vrc2(k) = K(3);
        debug.dOCV_dSOC(k) = dOCV_dSOC;
        debug.OCV_cc(k) = V_ocv_pred;
        debug.OCV_corr(k) = V_ocv_corr;
        debug.Rk_eff(k) = Rk_eff;
        debug.soc_gate_alpha(k) = soc_gate_alpha;
        debug.slope_factor(k) = slope_factor;
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

function d = distance_to_centroid(X, idx, C)
    n = size(X,1);
    d = nan(n,1);
    for i = 1:n
        k = idx(i);
        if k >= 1 && k <= size(C,1)
            d(i) = norm(X(i,:) - C(k,:));
        end
    end
end
%%
%% =========================================================
% FIGURE 2: PCA SHAPE (blue-black palette)
%% =========================================================
figure('Color','w','Name','PCA Shape-Only Residual Profiles');

blueblack = [
    0.05 0.05 0.05
    0.10 0.25 0.55
    0.20 0.45 0.80
    0.45 0.60 0.85
    0.65 0.75 0.90
    0.30 0.30 0.30
];

hold on;
for k = 1:bestK_shape
    idxk = CycleTable.cluster_shape == k;
    c = blueblack(mod(k-1,size(blueblack,1))+1,:);
    scatter(score_shape(idxk,1), score_shape(idxk,2), 28, ...
        'MarkerFaceColor', c, ...
        'MarkerEdgeColor', 'k', ...
        'DisplayName', sprintf('Cluster %d', k));
end
xlabel(sprintf('PC1 (%.1f%%)', explained_shape(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained_shape(2)));
title(sprintf('Shape-only residual profiles | Best K = %d', bestK_shape));
legend('Location','best');
grid on;
box on;
%%
%% =========================================================
% FIGURE 4: SHAPE-ONLY CLUSTER MEAN PROFILES (blue-black palette)
%% =========================================================
figure('Color','w','Name','Shape-Only Residual Mean Profiles');
tiledlayout(bestK_shape,1,'Padding','compact','TileSpacing','compact');

blueblack = [
    0.05 0.05 0.05
    0.10 0.25 0.55
    0.20 0.45 0.80
    0.45 0.60 0.85
    0.65 0.75 0.90
    0.30 0.30 0.30
];

for k = 1:bestK_shape
    nexttile;
    mu = MeanProfilesShape(k,:)';
    sd = StdProfilesShape(k,:)';
    c = blueblack(mod(k-1,size(blueblack,1))+1,:);

    fill([tau_grid; flipud(tau_grid)], [mu-sd; flipud(mu+sd)], ...
        c, 'EdgeColor', 'none', 'FaceAlpha', 0.18);
    hold on;
    plot(tau_grid, mu, '-', 'Color', c, 'LineWidth', 2.2);
    yline(0,'k:','LineWidth',1.0);

    xlabel('\tau');
    ylabel('Standardized residual');
    title(sprintf('Shape Cluster %d | n = %d', k, sum(CycleTable.cluster_shape==k)));
    grid on;
    box on;
end
%%
%% =========================================================
% FIGURE 7: MEAN RESIDUAL BY CYCLE LABEL
% only first / middle / last
%% =========================================================
figure('Color','w','Name','Mean Residual By Cycle Label');
hold on;

LabelKeep = ["first"; "middle"; "last"];
label_colors = [
    0.10 0.10 0.10   % first -> almost black
    0.10 0.30 0.65   % middle -> dark blue
    0.45 0.65 0.90   % last -> light blue
];

for i = 1:numel(LabelKeep)
    lab = LabelKeep(i);
    idx = CycleTable.cycle_label == lab;
    if ~any(idx), continue; end

    Xlab = ResidualRaw(idx, :);
    mu = mean(Xlab, 1, 'omitnan')';
    sd = std(Xlab, 0, 1, 'omitnan')';

    c = label_colors(i,:);

    fill([tau_grid; flipud(tau_grid)], [mu-sd; flipud(mu+sd)], ...
        c, 'EdgeColor', 'none', 'FaceAlpha', 0.12);
    plot(tau_grid, mu, 'LineWidth', 2.2, 'Color', c, 'DisplayName', char(lab));
end

%yline(0,'k:','LineWidth',1.0);
xlabel('\tau');
ylabel('Residual');
title('Mean EKF residual profile by cycle label');
legend('Location','best');
grid on;
box on;
%%
%% =========================================================
% FIGURE X: MEAN SOC ESTIMATION BY CYCLE LABEL
% horizontal subplots, solid lines only
% true SOC = light gray
% EKF SOC  = light blue (same for all cycle labels)
%% =========================================================
close(findobj('Type','figure','Name','Mean SOC Estimation By Cycle Label'));
figure('Color','w','Name','Mean SOC Estimation By Cycle Label');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

LabelKeep = ["first"; "middle"; "last"];

true_color = [0 0 0];   % light gray
ekf_color  = [0.45 0.65 0.90];   % light blue

for i = 1:numel(LabelKeep)
    lab = LabelKeep(i);
    idx = CycleTable.cycle_label == lab;
    if ~any(idx), continue; end

    mu_true = mean(SOCTrueProfiles(idx,:), 1, 'omitnan')';
    mu_est  = mean(SOCEstProfiles(idx,:),  1, 'omitnan')';

    nexttile;
    plot(tau_grid, mu_true*100, '--', 'Color', true_color, 'LineWidth', 1.2, ...
        'DisplayName', 'True SOC');
    hold on;
    plot(tau_grid, mu_est*100, '-', 'Color', ekf_color, 'LineWidth',1.2, ...
        'DisplayName', 'EKF Estimated SOC');

    xlabel('\tau');
    ylabel('SOC [%]');
    title(sprintf('%s cycles', char(lab)));
    legend('Location','best');
    grid on;
    box on;
end
%%
%% =========================================================
% FIGURE X: MEAN VOLTAGE TRACKING BY CYCLE LABEL
% horizontal subplots
% measured V = black dashed
% modelled V = light blue
%% =========================================================
close(findobj('Type','figure','Name','Mean Voltage Tracking By Cycle Label'));
figure('Color','w','Name','Mean Voltage Tracking By Cycle Label');
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

LabelKeep = ["first"; "middle"; "last"];

true_color = [0 0 0];
model_color = [0.45 0.65 0.90];

for i = 1:numel(LabelKeep)
    lab = LabelKeep(i);
    idx = CycleTable.cycle_label == lab;
    if ~any(idx), continue; end

    mu_trueV  = mean(VoltageTrueProfiles(idx,:),  1, 'omitnan')';
    mu_modelV = mean(VoltageModelProfiles(idx,:), 1, 'omitnan')';

    nexttile;
    plot(tau_grid, mu_trueV, '--', 'Color', true_color, 'LineWidth', 1.2, ...
        'DisplayName', 'Measured Voltage');
    hold on;
    plot(tau_grid, mu_modelV, '-', 'Color', model_color, 'LineWidth', 1.2, ...
        'DisplayName', 'Modelled Voltage');

    xlabel('\tau');
    ylabel('Voltage [V]');
    title(sprintf('%s cycles', char(lab)));
    legend('Location','best');
    grid on;
    box on;
end
%%

%% =========================================================
% COMBINED FIGURE: SOC ESTIMATION (TOP) + VOLTAGE TRACKING (BOTTOM)
%% =========================================================
close(findobj('Type','figure','Name','Combined SOC and Voltage Tracking'));
figure('Color','w','Name','Combined SOC and Voltage Tracking');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

LabelKeep = ["first"; "middle"; "last"];
LabelTitles = ["BoL"; "90% SOH"; "EoL"];

true_soc_color = [0.75 0.75 0.75];   % light gray
ekf_soc_color  = [0.45 0.65 0.90];   % light blue

true_v_color   = [0 0 0];            % black
model_v_color  = [0.45 0.65 0.90];   % light blue

% =========================
% Top row: SOC
% =========================
for i = 1:numel(LabelKeep)
    lab = LabelKeep(i);
    idx = CycleTable.cycle_label == lab;
    if ~any(idx), continue; end

    mu_true = mean(SOCTrueProfiles(idx,:), 1, 'omitnan')';
    mu_est  = mean(SOCEstProfiles(idx,:),  1, 'omitnan')';

    nexttile(i);
    plot(tau_grid, mu_true*100, '--', 'Color', true_soc_color, 'LineWidth', 1.5, ...
        'DisplayName', 'True SOC');
    hold on;
    plot(tau_grid, mu_est*100, '-', 'Color', ekf_soc_color, 'LineWidth', 1.8, ...
        'DisplayName', 'EKF Estimated SOC');

    xlabel('Normalized cycle time');
    ylabel('SOC [%]');
    title(LabelTitles(i));
    legend('Location','best');
    grid on;
    box on;
end

% =========================
% Bottom row: Voltage
% =========================
for i = 1:numel(LabelKeep)
    lab = LabelKeep(i);
    idx = CycleTable.cycle_label == lab;
    if ~any(idx), continue; end

    mu_trueV  = mean(VoltageTrueProfiles(idx,:),  1, 'omitnan')';
    mu_modelV = mean(VoltageModelProfiles(idx,:), 1, 'omitnan')';

    nexttile(i+3);
    plot(tau_grid, mu_trueV, '--', 'Color', true_v_color, 'LineWidth', 1.5, ...
        'DisplayName', 'Measured Voltage');
    hold on;
    plot(tau_grid, mu_modelV, '-', 'Color', model_v_color, 'LineWidth', 1.8, ...
        'DisplayName', 'Modelled Voltage');

    xlabel('Normalized cycle time');
    ylabel('Voltage [V]');
    title(LabelTitles(i));
    legend('Location','best');
    grid on;
    box on;
end
