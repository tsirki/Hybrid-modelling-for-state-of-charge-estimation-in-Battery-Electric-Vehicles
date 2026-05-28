%% preprocessing.m
% -------------------------------------------------------------------------
% Author: Konstantinos Tsirkinidis
% Date: 2025-11-07
% Article: Hybrid Modeling for State of Charge Estimation in 
% Battery Electric Vehicles: Balancing Accuracy, Efficiency, and 
% Interpretability
% Description:
%   Combines, cleans, and aligns three MATR battery dataset into a unified
%   structure ready for modeling
%
%   Input: MATR_batch_20170512.mat, MATR_batch_20170630.mat, and
% MATR_batch_20180412.mat
%   Output: Workspace variables with cleaned and resampled signals.
%
% Dependencies:
%   - MATLAB R2020a or newer
%
% -------------------------------------------------------------------------
%% Part_1: Loading the MATR dataset
% Initialize environment
clear; close all; clc
% Load and merge Batch 1 & 2
load('MATR_batch_20170512')

batch1 = batch; 
numBat1 = size(batch1,2);

load('MATR_batch_20170630')

add_len = [661, 980, 1059, 207, 481];
summary_var_list = {'cycle','QDischarge','QCharge','IR','Tmax','Tavg',...
    'Tmin','chargetime'};
batch2_idx = [8:10,16:17];
for i=1:5
    batch1(i).cycles(end+1:end+add_len(i)+1) = batch(batch2_idx(i)).cycles;
    batch1(i).summary.cycle(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.cycle;
    batch1(i).summary.QDischarge(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.QDischarge;
    batch1(i).summary.QCharge(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.QCharge;
    batch1(i).summary.IR(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.IR;
    batch1(i).summary.Tmax(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tmax;
    batch1(i).summary.Tavg(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tavg;
    batch1(i).summary.Tmin(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.Tmin;
    batch1(i).summary.chargetime(end+1:end+add_len(i)+1) = ...
        batch(batch2_idx(i)).summary.chargetime;
end

batch([8:10,16:17]) = [];
batch2 = batch;
numBat2 = size(batch2,2);
clearvars batch

% Clean and filter Batch 3
load('MATR_batch_20180412')
batch3 = batch;
batch3(38) = []; %remove channel 46 upfront; there was a problem with 
%the data collection for this channel
numBat3 = size(batch3,2);
endcap3 = zeros(numBat3,1);
clearvars batch
for i = 1:numBat3
    endcap3(i) = batch3(i).summary.QDischarge(end);
end
rind = find(endcap3 > 0.885);
batch3(rind) = [];

%remove the noisy Batch 8 batteries
nind = [3, 40:41];
batch3(nind) = [];
numBat3 = size(batch3,2);

batch_combined = [batch1, batch2, batch3];
numBat = numBat1 + numBat2 + numBat3;

%optionally remove the batteries that do not finish in Batch 1; depending
%on the modeling goal, you may not want to do this step
batch_combined([9,11,13,14,23]) = [];
numBat = numBat - 5;
numBat1 = numBat1 - 5; 

clearvars -except batch_combined numBat1 numBat2 numBat3 numBat
%% Part_2: Resampling & Cleaning
dt = 0.05;
resample_time = (0:dt:54.375)';
n_time = length(resample_time);
nan_threshold = 0.2;

% Initialize containers
I_all = {};
V_all = {};
T_all = {};
R0_all = {};
valid_cols_all = {};
Q_all = {};  % now Q_all stores time series (Q_cycle per cycle)

for battery_no = 1:length(batch_combined)
    cycles = batch_combined(battery_no).cycles;
    N = length(cycles) - 1;
    R0_battery = NaN(1, N);

    I_cleaned_cycles = {};
    V_cleaned_cycles = {};
    T_cleaned_cycles = {};
    t_cleaned_cycles = {};
    Q_cleaned_cycles = {};
    valid_cols = false(1, N);

    for i = 1:N
        try
            data = cycles(i+1);
            t_raw = data.t;
            I_raw = data.I;
            V_raw = data.V;
            T_raw = data.T;
            Qc_raw = data.Qc;  % time series
            Qd_raw = data.Qd;  % time series

            R0_battery = batch_combined(battery_no).summary.IR;

            if ~isempty(t_raw) && length(t_raw) > 1
                [t_unique, idx_unique] = unique(t_raw, 'stable');
                I_unique = I_raw(idx_unique);
                V_unique = V_raw(idx_unique);
                T_unique = T_raw(idx_unique);
                Qc_unique = Qc_raw(idx_unique);
                Qd_unique = Qd_raw(idx_unique);

                valid_idx = ~(isnan(t_unique) | isnan(I_unique) | isnan(V_unique) | isnan(T_unique) | ...
                              isnan(Qc_unique) | isnan(Qd_unique));
                t_final = t_unique(valid_idx);
                I_final = I_unique(valid_idx);
                V_final = V_unique(valid_idx);
                T_final = T_unique(valid_idx);
                Qc_final = Qc_unique(valid_idx);
                Qd_final = Qd_unique(valid_idx);

                if numel(t_final) > nan_threshold * (max(t_final) - min(t_final)) / dt
                    Q_cycle = Qc_final - Qd_final;

                    I_cleaned_cycles{end+1} = I_final;
                    V_cleaned_cycles{end+1} = V_final;
                    T_cleaned_cycles{end+1} = T_final;
                    t_cleaned_cycles{end+1} = t_final;
                    Q_cleaned_cycles{end+1} = Q_cycle;
                    valid_cols(i) = true;
                end
            end
        catch
            continue
        end
    end

    % Store per-battery results
    I_all{battery_no} = I_cleaned_cycles;
    V_all{battery_no} = V_cleaned_cycles;
    T_all{battery_no} = T_cleaned_cycles;
    t_all{battery_no} = t_cleaned_cycles;
    Q_all{battery_no} = Q_cleaned_cycles;
    R0_all{battery_no} = R0_battery;
    valid_cols_all{battery_no} = valid_cols;
end
