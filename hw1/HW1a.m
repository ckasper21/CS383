%
% Name: HW1a (CS 383) - Dimensionality Reduction via PCA
% Author: Chris Kasper
% Date: 1/16/19
%
clear; clc;
%%
% Lets import the data

dirName = 'yalefaces/*subject*';
files = dir(dirName);

D = [];

% Iterate through each file
for i = 1:size(files,1)
    f = strcat('yalefaces/',files(i).name);
    I = imread(f);
    % Down sample image to 40x40
    I = imresize(I,[40,40]);
    % Flatten image to 1x1600
    I = reshape(I,[1,numel(I)]);
    % Concatenate to Data matrix
    D = [D;I];
end

D = double(D);

%%
% Standardize Data

% Find mean
m = mean(D);
% Find std
s = std(D);

D = D - repmat(m,size(D,1),1);
D = D ./ repmat(s,size(D,1),1);

%%
% PCA

% Covarance of D
C = cov(D);

% Find eigenvectors and eigenvalues
[W,lam] = eig(C);
% Sort eigenvalues (to find the most relevant)
[~,idx] = sort(diag(lam),'descend');

% Want to reduce down to 2-D, so...
k = 2;
projMatrix = zeros([size(D,1),k]);

for i = 1:k
    % Z = XW
    %   W = eigenvector (in this case, the 1st or 2nd most relevant
    %   X = standardized data
    projMatrix(:,i) = D * W(:,idx(i));
end

%%
% Plot the data
x = projMatrix(:,1);
y = projMatrix(:,2);

scatter(x,y);
title('2D PCA Projection of Data');    