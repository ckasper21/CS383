%
% Name: HW3 (CS 383) - Linear Regression
% Author: Chris Kasper
% Date: 2/18/19
%
clear; clc; close all;
rng(0);
%% Gradient Descent - Fixed Learning Rate

termination = 2^-23;
iteration = 0;

% Set learning rate n
eta = 0.01;

% Initialize x to 0 and magn_change to 1 (for loop check purposes)
new_X = 0;
magn_change = 1;

% For plotting
plotX = [];

% Gradient Descent Calculation
while magn_change > termination
    iteration = iteration + 1;
    plotX = [plotX,new_X];
    
    gradientX = 4*(new_X-1)^3;
    old_X = new_X;
    new_X = old_X - (eta * gradientX);
    
    magn_change = abs(new_X - old_X);
    
end

% Plot
figure(1);
g_x = (plotX - 1).^4;
plotIteration = 1:1:iteration;
plot(plotIteration,g_x,'LineWidth',2);
title('Iterations vs g(x)');
xlabel('Iterations');
ylabel('g(x)');
axis([0 5000 -inf inf]);

figure(2);
plot(plotIteration,plotX,'LineWidth',2);
title('Iterations vs x');
xlabel('Iterations');
ylabel('x');
axis([0 5000 -inf inf]);

%% Gradient Descent - Adaptive Learning Rate
termination = 2^-23;
iteration_a = 0;

% Set learning rate n
eta_a = 1;

% Initialize x to 0 and magn_change to 1 (for loop check purposes)
new_X_a = 0;
magn_change_a = 1;

% For plotting
plotX_a = [];

% Have flag for changing learning rate
isPos = -1;

% Gradient Descent Calculation
while magn_change_a > termination
    iteration_a = iteration_a + 1;
    plotX_a = [plotX_a,new_X_a];
    
    gradientX_a = 4*(new_X_a-1)^3;
    
    % Check if gradient sign changed
    if gradientX_a > 0
        if isPos ~= 1
            isPos = 1;
            eta_a = eta_a/2;
        end
    else
        if isPos ~= 0
            isPos = 0;
            eta_a = eta_a/2;
        end
    end
    
    old_X_a = new_X_a;
    new_X_a= old_X_a - (eta_a * gradientX_a);
    
    magn_change_a = abs(new_X_a - old_X_a); 
end

% Plot
figure(3);
g_x_a = (plotX_a - 1).^4;
plotIteration_a = 1:1:iteration_a;
plot(plotIteration_a,g_x_a,'LineWidth',2);
title('Iterations vs g(x)');
xlabel('Iterations');
ylabel('g(x)');

figure(4);
plot(plotIteration_a,plotX_a,'LineWidth',2);
title('Iterations vs x');
xlabel('Iterations');
ylabel('x');

%% Closed Form Linear Regression

% Lets import the data
fileName = 'x06Simple.csv';

% Ignore first row and first column
D = csvread(fileName,1,1);

% Randomize Data using indices
indices = randperm(size(D,1));

% Grab first 2/3 of data for training data
trainingNum = round(size(D,1) * (2/3));
trainX = D(indices(1:trainingNum),1:(size(D,2)-1));
trainY = D(indices(1:trainingNum),end);

% Standardize Training Data
m = mean(trainX);
s = std(trainX);

trainX_s = trainX - repmat(m,size(trainX,1),1);
trainX_s = trainX_s ./ repmat(s,size(trainX,1),1);

% Add bias feature
trainX_s = [ones(1,size(trainX_s,1))',trainX_s];

% Compute coefficients
theta = (trainX_s'*trainX_s)^-1 * trainX_s'*trainY;

% Now grab the other 1/3 of data for testing
testX = D(indices(trainingNum+1:end),1:(size(D,2)-1));
testY = D(indices(trainingNum+1:end),end);

% Standarize Testing Data (using mean and std from training) 
% and add bias feature
testX_s = testX - repmat(m,size(testX,1),1);
testX_s = testX_s ./ repmat(s,size(testX,1),1);
testX_s = [ones(1,size(testX_s,1))',testX_s];

predictedTestY = testX_s * theta;

% RMSE (Root mean square error)
rmse = ((1/size(testX_s,1))*sum((testY - predictedTestY).^2))^.5;

%% S-Folds Cross-Validation

% Lets import the data
fileName = 'x06Simple.csv';

% Ignore first row and first column
D = csvread(fileName,1,1);

RMSEs = [];

% Lets do this 20 times
for i = 1:20

% Create S folds
S = 44;

% Randomize Data using indices
indices = randperm(size(D,1));
randData = D(indices,:);

% Length of S fold
s_len = round(size(randData,1) / S);

MSEs = [];

for j = 1:S
    % Get the begining and end of the S fold
    S_head = 1 + (j-1) * s_len;
    S_tail = min(S_head + s_len - 1, size(randData,1));
    
    % Adjust for unequal folds at the end
    if j == S
        S_tail = size(randData,1);
    end
    
    % Separate training and test data
    trainX = [randData(1:S_head-1, 1:(size(randData,2)-1)); randData(S_tail+1:end, 1:(size(randData,2)-1))];
    trainY = [randData(1:S_head-1, end); randData(S_tail+1:end, end)];
    
    testX = randData(S_head:S_tail,1:(size(randData,2)-1));
    testY = randData(S_head:S_tail,end);
    
    % Standardize training data and add bias feature
    m = mean(trainX);
    s = std(trainX);
    
    trainX_s = trainX - repmat(m,size(trainX,1),1);
    trainX_s = trainX_s ./ repmat(s,size(trainX,1),1);
    trainX_s = [ones(1,size(trainX_s,1))',trainX_s];
    
    % Compute coefficients
    theta = (trainX_s'*trainX_s)^-1 * trainX_s'*trainY;
    
    % Standarize Testing Data (using mean and std from training)
    % and add bias feature
    testX_s = testX - repmat(m,size(testX,1),1);
    testX_s = testX_s ./ repmat(s,size(testX,1),1);
    testX_s = [ones(1,size(testX_s,1))',testX_s];
    
    predictedTestY = testX_s * theta;
    
    thisMSE = (testY - predictedTestY).^2;
    MSEs = [MSEs; thisMSE];
end

thisRMSE = ((1/size(randData,1))*sum(MSEs))^.5;
RMSEs = [RMSEs; thisRMSE];
end

RMSEs_mean = mean(RMSEs);
RMSEs_std = std(RMSEs);

