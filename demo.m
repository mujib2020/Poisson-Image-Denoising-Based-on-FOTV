%% This demo tests FOTV regularized image denoising.
clear; close all; clc;

%% load a testing image
I = imread('barbara.png');
I = double(I); 


%% Set peak and add Poisson noise
peak = 55; I = I/max(I(:))*peak;
u0 = imnoise(uint8(I),'poisson');
u0 = double(u0);

%% FOTV
alpha = 1.6; beta = 12; mu = 2e-3; mu = mu*beta;
tic
[uFOTV,output1] = FOTV_denoising_ADMM(u0,beta,mu,alpha);
toc


%% TV
alpha = 1;
tic
[uTV,output2] = FOTV_denoising_ADMM(u0,beta,mu,alpha);
toc


%% calculate the PSNR values
psnr_input = PSNR(I,u0);
psnr_FOTV = PSNR(I,uFOTV);
psnr_TV = PSNR(I,uTV);

% plot the results 
figure;
subplot(221); imshow(I,[0,peak]); title('original')

subplot(222); imshow(u0,[0,peak]); 
title(sprintf('Noisy input (PSNR=%4.2f)',psnr_input))

subplot(223); imshow(uTV,[0,peak]); 
title(sprintf('TV (PSNR=%4.2f)',psnr_TV))

subplot(224); imshow(uFOTV,[0,peak]); 
title(sprintf('FOTV (PSNR=%4.2f)',psnr_FOTV))
