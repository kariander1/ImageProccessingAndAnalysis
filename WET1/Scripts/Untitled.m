%% sec. 2A 
clear

% Loading picture
im_path         = '..\circ.png';
im              = imread(im_path);
grayscale_im    = rgb2gray(im);

% Display both images
figure
imshowpair(im,grayscale_im,'montage');

%% sec. 2B

% Apply 2D-DFT
im_fft      = fft2(grayscale_im);
im_shifted  = fftshift(im_fft);

% Display log for convenience
im_logged   = log(1+abs(im_shifted));

% Display orig and transform images
subplot(1,2,1);
imshow(grayscale_im,[]);
title('Original Image')

subplot(1,2,2);
imshow(im_logged,[]);
title('Discrete 2D Fourier Transform')