 clear all;clc
%% Load data
D=load('matlab.mat');
A=D.datas(:,1:3);             % Inputs
B=D.datas(:,4);               % Targets
%% define Options
Opts.ELM_Type='Class';    % 'Class' for classification and 'Regrs' for regression
Opts.number_neurons=1000;  % Maximam number of neurons 
Opts.Tr_ratio=0.70;       % training ratio
Opts.Bn=1;                % 1 to encode  lables into binary representations
                          % if it is necessary
%% Training
[net]= elm_LB(A,B,Opts);
net
%% prediction
[output]=elmPredict(net,A);