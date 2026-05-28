%% =========================================================
% EXTRACT Q_nom_init FROM FIRST CYCLE OF EACH BATTERY
% Q_nom_init_per_battery(b) = max(Q_all{b}{1})
%% =========================================================

nB = numel(Q_all);

Q_nom_init_per_battery = nan(nB,1);
has_first_cycle = false(nB,1);

for b = 1:nB
    try
        if isempty(Q_all{b})
            continue;
        end

        if numel(Q_all{b}) < 1 || isempty(Q_all{b}{1})
            continue;
        end

        q1 = Q_all{b}{1};
        q1 = q1(:);
        q1 = q1(isfinite(q1));

        if isempty(q1)
            continue;
        end

        Q_nom_init_per_battery(b) = max(q1);
        has_first_cycle(b) = true;

    catch
        continue;
    end
end

QnomTable = table( ...
    (1:nB)', ...
    has_first_cycle, ...
    Q_nom_init_per_battery, ...
    'VariableNames', {'battery_no','has_first_cycle','Q_nom_init_first_cycle'} );

disp(QnomTable);

writetable(QnomTable, 'Q_nom_init_first_cycle_all_batteries.csv');
save('Q_nom_init_first_cycle_all_batteries.mat', ...
     'Q_nom_init_per_battery', 'QnomTable');
