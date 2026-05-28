clear; close all; clc

load('MATR_batch_20170512')

batch1 = batch; 
numBat1 = size(batch1,2);

load('MATR_batch_20170630')

%Some batteries continued from the first run into the second. We append 
%those to the first batch before continuing.
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

%%

% Initialize arrays
time = []; current = []; voltage = []; temperature = [];
for j = 1:length(data_from_cycles)
    time = [time; data_from_cycles(j).t];
    current = [current; data_from_cycles(j).I];
    voltage = [voltage; data_from_cycles(j).V];
    temperature = [temperature; data_from_cycles(j).T];
end
% Truncate or pad to 1087
    len = min(length(time), max_length);
 time_all(1:len, i) = time(1:len);
    current_all(1:len, i) = current(1:len);
    voltage_all(1:len, i) = voltage(1:len);
    temperature_all(1:len, i) = temperature(1:len);

%% Create 
% Remove duplicate timestamps
[time_unique, idx_unique] = unique(time);
current_unique = current(idx_unique);
voltage_unique = voltage(idx_unique);
temperature_unique = temperature(idx_unique);

% Sort by time (in case data is out of order)
[time_sorted, sort_idx] = sort(time_unique);
current_sorted = current_unique(sort_idx);
voltage_sorted = voltage_unique(sort_idx);
temperature_sorted = temperature_unique(sort_idx);


dt = 0.05; % Desired time step (e.g., 1 second)
t_uniform = (0.05:dt:max(time_unique))'; % Uniform time grid

% Interpolate signals to the uniform grid
%current_resampled = interp1(time_sorted, current_sorted, t_uniform, 'linear', 'extrap');
%voltage_resampled = interp1(time_sorted, voltage_sorted, t_uniform, 'linear', 'extrap');
%temperature_resampled = interp1(time_sorted, temperature_sorted, t_uniform, 'linear', 'extrap');

% Plot to verify
figure;
plot(t_uniform, current_sorted, 'b.', 'DisplayName', 'Raw Data'); 
hold on;
%plot(t_uniform, current_resampled, 'r-', 'DisplayName', 'Resampled');
xlabel('Time (s)'); ylabel('Current (A)'); 
%legend; title('Raw vs. Resampled Data');
%% Coulomb Counting
Q_nominal = 1.1; % in Ah
dt = 1; % seconds
SOC = zeros(size(current_sorted));
SOC(1) = 80; % Start at 100% SoC

for k = 2:length(current_sorted)
    SOC(k) = SOC(k-1) + (current_sorted(k) * dt) / (Q_nominal * 3600) * 100;
end

SOC = max(min(SOC, 100), 0); % Clamp between 0 and 100%

figure;
plot(t_uniform, SOC, 'g-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('SoC (%)');
title('State of Charge Over Time');
grid on;

%%
data = struct();
data.time = t_uniform;
data.current = current_resampled;
data.voltage = voltage_resampled;
data.temperature = temperature_resampled;
data.SOC = SOC;

save('preprocessed_battery_data.mat', 'data');


