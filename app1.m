clc;
clear;
close all;

%% Load Data

load data;
X=data.x;         % Inputs
Y=data.y;      % Targets

%% KNN Classifier

t=CTree.fit(X,Y);

disp('Resub. Loss =');
disp(resubLoss(t));

%% Cross-validation

cvmodel=crossval(t);

disp('k-Fold Loss =');
disp(kfoldLoss(cvmodel));

