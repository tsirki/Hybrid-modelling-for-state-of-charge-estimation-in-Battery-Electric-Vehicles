%% =========================================================
% CREATE COMPACT FUSION CONFIG WORKSPACE
%
% This script recreates the compact configuration MAT file used by the
% downstream EKF-GPR scripts. It intentionally does not save raw battery
% time-series data; those are produced by preprocessing.m and can be kept in
% the active workspace or in battery_workspace_core.mat.
%
% OUTPUT:
%   fusion_full_model.mat
%% =========================================================

% Original base training split used before the later Hybrid EKF-GPR expansion.
train_batteries = [ ...
    2 4 6 8 9 10 12 14 16 18 19 21 23 25 27 29 31 33 35 37 39 41];

% Legacy scalar nominal-capacity initialization retained for older scripts.
% Battery-specific initial capacities are generated separately by
% extract_qnom_init_first_cycle.m as Q_nom_init_per_battery.
Q_nom_init = 1.1 * 3600;

% Measurement noise settings used in the EKF residual workflows.
I_noise_std = 0.05;
V_noise_std = 0.03;

% Two-RC Thevenin model parameters used throughout the final MATLAB scripts.
R0 = 0.0167;
R1 = 0.01;
C1 = 3000;
R2 = 0.03;
C2 = 2000;

save('fusion_full_model.mat', ...
    'train_batteries', ...
    'Q_nom_init', ...
    'I_noise_std', 'V_noise_std', ...
    'R0', 'R1', 'C1', 'R2', 'C2');

fprintf('Saved fusion_full_model.mat with compact EKF/GPR configuration.\n');
