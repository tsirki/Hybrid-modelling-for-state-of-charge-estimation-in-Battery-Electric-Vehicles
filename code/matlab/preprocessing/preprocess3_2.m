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

%if ~isempty(I_cleaned_cycles)
   % save(sprintf('battery_%03d_preprocessed.mat', battery_no), ...
      %   't_cleaned_cycles', 'I_cleaned_cycles', 'V_cleaned_cycles', 'T_cleaned_cycles','R0_battery');
%end
%% 

% === Script Configuration (Assumes T_all exists in workspace) ===
batteries_to_check = [1, 10, 50, 100, 124];
Q_nominal = 1.1;  % Nominal Capacity in Ah

for idx = 1:length(batteries_to_check)
    battery_no = batteries_to_check(idx);
    
    % NOTE: Assumes 'valid_cols_all' is defined and loaded
    valid_cols = valid_cols_all{battery_no}; 
    if isempty(valid_cols)
        fprintf("Battery %d skipped (no valid cycles).\n", battery_no);
        continue
    end
    
    % Take first valid cycle
    valid_cycles_idx = find(valid_cols);
    cycle_idx = valid_cycles_idx(1);
    
    % === Load Data ===
    t_clean = t_all{battery_no}{cycle_idx};
    I_clean = I_all{battery_no}{cycle_idx};
    V_clean = V_all{battery_no}{cycle_idx};
    % 🔥 NEW: Load Temperature Data (Assumes T_all exists)
    T_clean = T_all{battery_no}{cycle_idx}; 

    % === SOC Calculation (based on raw integration) ===
    SOC_raw = cumtrapz(t_clean, I_clean) / (Q_nominal * 3600);
    SOC_norm = (SOC_raw - min(SOC_raw)) / (max(SOC_raw) - min(SOC_raw));
    
    % === Print Info (with Temperature Range) ===
    fprintf("\nBattery %03d - Cycle %d VALIDATION:\n", battery_no, cycle_idx);
    fprintf("  Total Duration: %.2f min\n", max(t_clean) / 60);
    fprintf("  Mean Current: %.4f A\n", mean(I_clean));
    fprintf("  Temperature Range: [%.2f, %.2f] degC\n", min(T_clean), max(T_clean)); % NEW
    fprintf("  Data Points: %d\n", numel(t_clean));
    
    % === Plotting (Now 4 Subplots) ===
    figure;
    
    % 1. Current
    subplot(4,1,1); % Changed to 4 rows
    plot(t_clean, I_clean, 'k-');
    title(sprintf('Battery %d, Cycle %d - Data Summary', battery_no, cycle_idx));
    ylabel('Current [A]');
    grid on;
    
    % 2. Temperature 🔥 NEW SUBPLOT 🔥
    subplot(4,1,2); % Changed to 4 rows, this is the 2nd plot
    plot(t_clean, T_clean, 'm-');
    ylabel('Temp [°C]');
    grid on;
    
    % 3. Voltage
    subplot(4,1,3); % Changed to 4 rows, this is the 3rd plot
    plot(t_clean, V_clean, 'b-');
    ylabel('Voltage [V]');
    grid on;
    
    % 4. SOC
    subplot(4,1,4); % Changed to 4 rows, this is the 4th plot
    plot(t_clean, SOC_norm * 100, 'r-'); % Display SOC as percentage
    ylabel('Normalized SOC [%]');
    xlabel('Time [min]');
    grid on;
end
