%% Load data
data = readtable('datas-2.csv');
A = table2array(data(:, 2:4));   % Distance, Charge per delay, Scaled distance
B = table2array(data(:, 5));     % PPV

%% Define Options
Opts.ELM_Type = 'Class';  % Regression
Opts.number_neurons = 1000;  % Maximum number of neurons
Opts.Tr_ratio = 0.70;   % Training ratio
Opts.Bn = 1;

%% Training
[net] = elm_LB(A, B, Opts);
net

%% Prediction
output = elmPredict(net, A);

%% GWO Optimization
% Objective function
obj_func = @(x) elm_LB(A, B, struct('ELM_Type', 'Class', 'number_neurons', round(x(1)), 'Bn', round(x(2)), 'Tr_ratio', 0.7));

% Search space
lb = [10, 0]; % Lower bounds
ub = [10000, 2]; % Upper bounds

%% Plot predicted and ground truth values
figure;
plot(1:length(B), B, 'ro', 1:length(output), output, 'b*');
xlabel('Data Point');
ylabel('PPV (mm/s)');
legend('Ground Truth', 'Predicted');
title(sprintf('Mean Absolute Error: %.2f', mae));

