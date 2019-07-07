%
% Name: HW1b (CS 383) - Eigenfaces
% Author: Chris Kasper
% Date: 1/16/19
%
clear; clc;
close all;
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
backup = D;
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

% % Most important principle component (primary principle component)
ppc = W(:,idx(1));
im_ppc = flip(reshape(ppc,[40,40]),2);

figure(1);
imshow(im_ppc, [min(ppc) max(ppc)])
title('Primary Principle Component');

% Find k to encode at least 95% of the information
alpha = .95;
n_lam = diag(lam);
sum_lam = sum(abs(n_lam));
sum_lam_k = 0;

for i = 1:size(diag(lam),1)
    sum_lam_k = sum_lam_k + n_lam(idx(i));
    
    if (sum_lam_k / sum_lam) >= alpha
        break
    end
end
k = i;
fprintf('Number of ks to use = %i\n', k);

% Get the original image
D_o = backup(1,:);
im_D_o = reshape(D_o, [40,40]);

% Reconstruct the original image using just ppc
D_ppc = D(1,:);
Z_ppc = D_ppc * ppc;
x_ppc = Z_ppc * ppc';

im_x_ppc = reshape(x_ppc, [40,40]);

% Reconstruct the orignal image using k
W_k = [];
for i = 1:k
    W_k = [W_k W(:,idx(i))]; 
end

D_k = D(1,:);
Z_k = D_k * W_k;
x_k = Z_k * W_k';

im_x_k = reshape(x_k, [40,40]);

% Show each image for reconstruction
figure(2);
subplot(1,3,1);
imshow(im_D_o);
title('Original Image');
subplot(1,3,2);
imshow(im_x_ppc, [min(x_ppc),max(x_ppc)]);
title('Single PC Reconstruction');
subplot(1,3,3);
imshow(im_x_k, [min(x_k),max(x_k)]);
title('k PC Reconstruction');


