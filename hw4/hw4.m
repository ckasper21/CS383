%
% Name: HW4 (CS 383) - Classification
% Author: Chris Kasper
% Date: 2/25/19
%
clear; clc; close all;

%% Naive Bayes Classifier
rng(0);

% Lets import the data
fileName = 'spambase.data';
D = csvread(fileName);

% Randomize Data using indices
indices = randperm(size(D,1));
randData = D(indices,:);

% Make first 2/3 of data the training set
trainingNum = round(size(D,1) * (2/3));
trainX = randData(1:trainingNum,1:(size(D,2)-1));
trainY = randData(1:trainingNum,end);

% Standardize training set
m = mean(trainX);
s = std(trainX);

trainX_s = (trainX - m) ./ s;

% Divide training data into c0 and c1 (classes)
% (with respect to spamdata.base, c0 = not spam, c1 = spam)
c0 = find(trainY==0);
c1 = find(trainY==1);

% Create models for c0 (not spam) for each feature
c0_models = [];

for i = 1:size(trainX_s,2)
    thisFeature = trainX_s(c0,i);
    
    thisMean = mean(thisFeature);
    thisStd = std(thisFeature);
    c0_models = [c0_models; [thisMean, thisStd]];
end

% Create models for c1 (spam) for each feature
c1_models = [];

for i = 1:size(trainX_s,2)
    thisFeature = trainX_s(c1,i);
    
    thisMean = mean(thisFeature);
    thisStd = std(thisFeature);
    c1_models = [c1_models; [thisMean, thisStd]];
end

% Set up test data (it is the last 1/3 of data)
testX = randData(trainingNum+1:end,1:(size(D,2)-1));
testY = randData(trainingNum+1:end,end);

% Standardize it
testX_s = (testX - m) ./ s;

% Classify test samples using the c0 and c1 models
predictedClass = [];

prob_c0 = sum(trainY==0)/size(trainY,1);
prob_c1 = sum(trainY==1)/size(trainY,1);

for i = 1:size(testX_s,1)
    thisSample = testX_s(i,:);
    
    % Get probability of each class
    p0 = (1 ./ (c0_models(:,2) .* (2*pi)^.5)) .* exp(-(thisSample - c0_models(:,1)).^2 ...
        ./ (2 .* c0_models(:,2).^2));
    p0 = diag(p0);
    
    p1 = (1 ./ (c1_models(:,2) .* (2*pi)^.5)) .* exp(-(thisSample - c1_models(:,1)).^2 ...
        ./ (2 .* c1_models(:,2).^2));
    p1 = diag(p1);
    
    % Calculate total probability
    total_p0 = prob_c0;
    total_p1 = prob_c1;
    
    for j = 1:size(p0,1)
        total_p0 = total_p0 * p0(j);
        total_p1 = total_p1 * p1(j);
    end
    
    % Assign sample to class with higher total probability
    if total_p0 > total_p1
        predictedClass = [predictedClass;0];
    else
        predictedClass = [predictedClass;1];
    end
   
end

% Find number of true positive, true negative, false positive, false
% negative
TP = 0;
TN = 0;
FP = 0;
FN = 0;

for i = 1:size(testY,1)
    if testY(i) == 1
        if predictedClass(i) == 1
            TP = TP + 1;
        else
            FN = FN + 1;
        end
    else
        if predictedClass(i) == 1
            FP = FP + 1;
        else
            TN = TN + 1;
        end 
    end
end

% Compute statistics
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f_measure = (2 * precision * recall) / (precision + recall);
accuracy = (TP + TN) / (TP + TN + FP + FN);

fprintf('Precison = %.4f\n',precision);
fprintf('Recall = %.4f\n',recall);
fprintf('F-Measure = %.4f\n',f_measure);
fprintf('Accuracy = %.4f\n',accuracy);