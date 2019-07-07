%
% Name: HW2 (CS 383) - Clustering
% Author: Chris Kasper
% Date: 1/29/19
%
clear; clc; close all;
%%
% Lets import the data

fileName = 'diabetes.csv';

D = csvread(fileName);

% Separate class label from data
% Y = Class labels, X = Features
Y = D(:,1);
X = D(:,2:end);

% Standardize Data

% Find mean
m = mean(X);
% Find std
s = std(X);

X_s = X - repmat(m,size(X,1),1);
X_s = X_s ./ repmat(s,size(X,1),1);

% Seed random generator
rng(0);

% Test runs
d1 = X_s(:,1:2);
[v1, F] = myKMeans(d1,Y,2);
%createVideo(F,'K_2_F_12.avi');

d2 = X_s(:,1:2);
[v2, F] = myKMeans(d2,Y,3);
%createVideo(F,'K_3_F_12.avi');

d3 = X_s(:,1:4);
[v3, F] = myKMeans(d3,Y,4);
%createVideo(F,'K_4_F_1234.avi');

%% Define myKMeans 
% inputs:   data = observable data
%           labels = class labels with data
%           numClusters = number of clusters
% output:   v = reference vectors for each cluster
%           F = frames (for video)

function [v,F] = myKMeans(data,labels,numClusters)

dims = size(data,2);

colors = ['b','r','g','m','c','y','k'];

% Restriction based on assignment
if (numClusters < 1) && (numClusters > 7)
    fprintf('numClusters not in range from 1 to 7');
    return;
end

if (dims == 2) || (dims == 3)
    fprintf('No PCA needed\n');
    D = data;
    
elseif dims > 3
    fprintf('Need PCA\n');
    
    % Get covariance matrix
    C = cov(data);
    
    % Find eigenvectors and eigenvalues
    [W,lam] = eig(C);
    % Sort eigenvalues (to find the most relevant)
    [~,idx] = sort(diag(lam),'descend');
    
    % Want to reduce down to 3-D, so...
    k = 3;
    projMatrix = zeros([size(data,1),k]);
    
    for i = 1:k
        % Z = XW
        %   W = eigenvector (in this case, the 1st or 2nd most relevant
        %   X = standardized data
        projMatrix(:,i) = data * W(:,idx(i));
    end
    
    dims = 3;
    D = projMatrix;
    
end

% Randomly select initial reference vectors by on numClusters (k)
v = zeros(numClusters,size(D,2));
randIdx = randperm(size(D,1));

for i = 1:numClusters
    v(i,:) = D(randIdx(i),:);
end

% Start kMeans
iteration = 0;

% Cluster Assignments
c = zeros(1,size(D,1));

% Termination factor
eps = 2^-23;

% Magnitude of cluster change
magCC = 0;
prev_v = 0;

figure;

if dims == 2
    
    while (magCC > eps) || (iteration == 0)
        clf;
        
        iteration = iteration + 1;
        prev_v = v;
        
        % Start assigning observations to clusters
        for i = 1:size(D,1)
            x = D(i,:);
            dists = sqrt((v(:,1)-x(1)).^2 + (v(:,2)-x(2)).^2);
            shortDistIdx = find(dists == min(dists));
            c(i) = shortDistIdx(1);
        end
        
        sumClusterQuants = 0;
        
        % Assign new reference vectors
        for j = 1:numClusters
            idxC = c==j;
            sumC = sum(D(idxC,:));
            numInC = nnz(idxC);
            v(j,:) = sumC / numInC;
            
            % Compute purity for this cluster
            clusterLabels = labels(idxC);
            uniqueLabels = unique(clusterLabels);
            
            thisClusterPurity = 0;
            
            maxQuants = [];
            for k = 1:size(uniqueLabels,1)
                idxL = clusterLabels == uniqueLabels(k);
                maxQuants = [maxQuants,nnz(idxL)];
            end
            
            thisClusterPurity = max(maxQuants) / numInC;
            
            % Add this max cluster quantity to the sum of of quantities
            % This is for total purity calculation
            sumClusterQuants = sumClusterQuants + max(maxQuants);
            
            scatter(D(idxC,1),D(idxC,2),colors(j),'x');
            hold on;
            scatter(v(j,1),v(j,2),100,colors(j),'o','filled','MarkerEdgeColor','k');
            hold on;
            
        end
        
        % Calculate overall purity
        totalPurity = sumClusterQuants / size(data,1);
        
        % Put purity on graph
        title(['Iteration #' num2str(iteration), ' | Total Purity = ' ...
            num2str(totalPurity)]);
        
        F(iteration) = getframe(gcf);
        pause(2);
        
        % Calculate magnitude of cluster change
        magCC = sum(sqrt((v(:,1)-prev_v(:,1)).^2 + (v(:,2)-prev_v(:,2)).^2));
    end
    
elseif dims == 3
    
    rotate3d on;
    
    while (magCC > eps) || (iteration == 0)
        clf;
        
        iteration = iteration + 1;
        prev_v = v;
        
        % Start assigning observations to clusters
        for i = 1:size(D,1)
            x = D(i,:);
            dists = sqrt((v(:,1)-x(1)).^2 + (v(:,2)-x(2)).^2 + ...
                (v(:,3)-x(3)).^2);
            shortDistIdx = find(dists == min(dists));
            c(i) = shortDistIdx(1);
        end
        
        sumClusterQuants = 0;
        
        % Assign new reference vectors
        for j = 1:numClusters
            idxC = c==j;
            sumC = sum(D(idxC,:));
            numInC = nnz(idxC);
            v(j,:) = sumC / numInC;
            
            % Compute purity for this cluster
            clusterLabels = labels(idxC);
            uniqueLabels = unique(clusterLabels);
            
            thisClusterPurity = 0;
            
            maxQuants = [];
            for k = 1:size(uniqueLabels,1)
                idxL = clusterLabels == uniqueLabels(k);
                maxQuants = [maxQuants,nnz(idxL)];
            end
            
            thisClusterPurity = max(maxQuants) / numInC;
            
            % Add this max cluster quantity to the sum of of quantities
            % This is for total purity calculation
            sumClusterQuants = sumClusterQuants + max(maxQuants);
            
            scatter3(D(idxC,1),D(idxC,2),D(idxC,3),colors(j),'x');
            hold on;
            scatter3(v(j,1),v(j,2),v(j,3),100,colors(j),'o','filled','MarkerEdgeColor','k');
            hold on;
            
        end
        
        % Calculate overall purity
        totalPurity = sumClusterQuants / size(data,1);
        
        % Put purity on graph
        title(['Iteration #' num2str(iteration), ' | Total Purity = ' ...
            num2str(totalPurity)]);
        
        F(iteration) = getframe(gcf);
        pause(2);
        
        % Calculate magnitude of cluster change
        magCC = sum(sqrt((v(:,1)-prev_v(:,1)).^2 + (v(:,2)-prev_v(:,2)).^2) + ...
            (v(:,3)-prev_v(:,3)).^2);
    end
    
end

end

function createVideo(F,strName)

writerObj = VideoWriter(strName);

% set the frames per second
writerObj.FrameRate = 1;

% open the video writer
open(writerObj);

% write the frames to the video
for i=1:length(F)
    % convert the image to a frame
    frame = F(i) ;
    writeVideo(writerObj, frame);
end

% close the writer object
close(writerObj);

end
