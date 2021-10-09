
%% sec. 1A

% Load video:
vid_path        = '../Corsica.mp4';
video           = VideoReader(vid_path);
implay(vid_path); %Play Video

%% sec. 1B
% Extract fps
fps             = get(video,'FrameRate');
requested_sec   = 4;

% Get the 101st frame (first frame of 4th second)
requested_frame    = read(video,fps*requested_sec+1); 
grayscale_frame = rgb2gray(requested_frame); % Get grayscale representation from RGB


imhist(grayscale_frame);    % Create grayscale histogram of the frame
figure                      % Create new figure for image
imshow(grayscale_frame);    % Display image

%% sec. 1C Gamma Correction

% Apply gamma corrections and save
gamma_half          = imadjust(grayscale_frame,[],[],0.5);
gamma_one_and_half  = imadjust(grayscale_frame,[],[],1.5);

% Plot pairs of images
imshowpair(gamma_half,gamma_one_and_half,'montage');
title('γ = 0.5                                                                                                        γ = 1.5')

subplot(1,2,1);


% Plot histograms
imhist(gamma_half);
title('γ = 0.5')
subplot(1,2,2);

imhist(gamma_one_and_half);
title('γ = 1.5')

%% sec. 1E 1 Second sampling

avg_mean_video(video,263,1); % Display median & mean of 4:23-4:24


%% sec. 1F 3 Second sampling

avg_mean_video(video,261,3); % Display median & mean of 4:21-4:24


%% sec. 2A 
clear

% Loading picture
im_path         = '..\building.jpg';
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

%% sec. 2C
[M , N]=size(im_fft); % image size

% Create filter for 2% horizontal frequencies
low_pass_horizontal_filter = zeros(M,N);                % Create template of zeros
low_pass_horizontal_filter(M/2-0.02*M:M/2+0.02*M,:)=1;  % Take middle of image +- 2% and insert ones

% Create filter for 2% vertical frequencies
low_pass_vertical_filter = zeros(M,N);                  % Create template of zeros
low_pass_vertical_filter(:,N/2-0.02*N:N/2+0.02*N)=1;    % Take middle of image +- 2% and insert ones

% Merge two filters to get third filter
low_pass_cross_filter = low_pass_horizontal_filter | low_pass_vertical_filter;

% Apply functions
display_filtered(grayscale_im,low_pass_vertical_filter,'2% I direction low freq.',true);
display_filtered(grayscale_im,low_pass_horizontal_filter,'2% K direction low freq.',true);
reversed_im_cross = display_filtered(abs(grayscale_im),low_pass_cross_filter,'2% I & K direction low freq.',true);

%% sec. 2D

% Created function for displaying dominant frequencies
display_dominant(grayscale_im,0.1,true);

%% sec. 2E

reversed_im = display_dominant(grayscale_im,0.02,true);

% Compare dominant frequencies to 2% low frequencies from section 2C.
figure
subplot(1,3,1);
imshow(grayscale_im,[]);
title("Original Image")


subplot(1,3,2);
imshow(abs(reversed_im_cross),[]);
title("2% I & K direction low freq.")

subplot(1,3,3);
imshow(reversed_im,[]);
title("2% Dominant Frequencies")

%% sec. 2F

% Create array of zeros
mse_vec = zeros(100,1);

for p = 1:100
    % Save the reconstructed image and populate the vector with the MSE
    reversed_im = display_dominant(grayscale_im,p/100,false);
    mse_vec(p) = immse(reversed_im,double(grayscale_im));
end

figure
plot(mse_vec)
ylabel("Mean Square Error")
xlabel("P [%]")
title('Mean Square Error as function of P')

%% sec. 3A
clear

% Load selfie and parrot
parrot_path     = '..\parrot.png';
im_path         = '..\yours.jpg';
im_parrot      = rgb2gray(imread(parrot_path));

[M,N] = size(im_parrot);

% Resize the selfie (which is amazing) to parrot dimensions
im      = imresize( rgb2gray(imread(im_path)),[M , N] );
imshowpair(im_parrot,im,'montage');
title('Parrot                                                                Selfie')

%% sec. 3B

% Create 2D-DFT
im_fft      = fftshift(fft2(im));
parrot_fft  = fftshift(fft2(im_parrot));

% Create amp & phase images of DFT of selfie
amplitude_im        = abs(im_fft);
phase_im            = angle(im_fft);

% Create amp & phase images of DFT of parrot
amplitude_parrot    = abs(parrot_fft);
phase_parrot        = angle(parrot_fft);

% Plot images
subplot(1,2,2);
imshow(log(1+amplitude_im),[]);
title('Amplitude yours.jpg')

subplot(1,2,1);
imshow(log(1+amplitude_parrot),[]);
title('Amplitude parrot')

%% sec. 3C

% Switch phase and amp. between images
new_yours_fft   = amplitude_im .* exp(j * phase_parrot);
new_parrot_fft  = amplitude_parrot .* exp(j * phase_im);

new_yours       = ifft2(ifftshift(new_yours_fft));
new_parrot      = ifft2(ifftshift(new_parrot_fft));

% Plot images
subplot(1,2,1);
imshow(new_yours,[]);
title('yours amp & parrot phase')

subplot(1,2,2);
imshow(new_parrot,[]);
title('parrot amp & yours phase')

%% sec. 3D

% Get range for random amp. (min and max of existing amplitudes)
min_amp = min(amplitude_im,[],'all');
max_amp = max(amplitude_im,[],'all');

% Create random matrix with values within the range [0,1] and multiply by
% the difference of max and min amp values, and add the minimum amp available.
% This way we assure the randomized values are always within [min_amp,max_amp]
rnd_amp = min_amp+ rand(size(amplitude_im)) * (max_amp - min_amp);

% Reconstruct image
new_im       = ifft2(ifftshift(rnd_amp .* exp(j * phase_im)));

imshow(abs(new_im),[]);
title('Reconstructed image')

%% sec. 4D

% Get range for random amp. (min and max of existing phases)
min_p = min(phase_im,[],'all');
max_p = max(phase_im,[],'all');

% Create random matrix with values within the range [0,1] and multiply by
% the difference of max and min phase values, and add the minimum phase available.
% This way we assure the randomized values are always within [min_p,max_p]
rnd_p = min_p+ rand(size(phase_im)) * (max_p - min_p);

% Reconstruct image
new_im       = ifft2(ifftshift(amplitude_im .* exp(j * rnd_p)));

imshow(abs(new_im),[]);
title('Reconstructed image')


function im_reversed = display_dominant(image, percentage,display)
    
    % Get 2D-DFT of image
    im_fft      = fft2(image);
    im_shifted  = fftshift(im_fft);
    
    % Get unique amplitudes of image
    amp_im      = abs(im_shifted);   
    unique_amps = unique(amp_im);

    % Calculate the index corresponding to requested percentage
    amp_threshold_index = round(1+(1-percentage)*size(unique_amps,1));
    
    % Get the threshold amplitude for filtering
    amp_threshold       = unique_amps(amp_threshold_index);

    % Create boolean matrix for filtering
    im_filter   = (amp_im >= amp_threshold);


    % Call display_filtered and save reconstructed image for return
    im_reversed = display_filtered(image,im_filter,strcat(string(percentage*100),'% Dominant Frequencies'),display);
end
function im_reversed = display_filtered(image, filter,title_name,display)

    % Get 2D-DFT of image
    im_fft      = fft2(image);
    im_shifted  = fftshift(im_fft);
    
    % Apply filter by dot product
    im_fourier_filtered = im_shifted .* filter;

    % Reconstruct image
    im_unshifted = ifftshift(im_fourier_filtered);
    im_reversed = ifft2(im_unshifted );
    
    % Display reconstruction if requested
    if display
        
        % Plot graphs
        figure
        subplot(1,3,1);

        imshow(log(1+abs(im_fourier_filtered)),[]);
        title('Shifted fft2')

        subplot(1,3,2);
        imshow(log(1+abs(im_unshifted)),[]);
        title('Un-shifted fft2')

        subplot(1,3,3);
        imshow(abs(im_reversed),[]);
        title(title_name);
    end
end




%% sec. 1D Sampling of video + generic function for displaying mean and median
function [mean_vid,median_vid] = avg_mean_video(video,selected_second, duration_sec)
    
    fps             = get(video,'FrameRate');
    % Extract frames from video
    one_sec_vid     = read(video, [ (fps*selected_second +1)  (fps*selected_second +duration_sec*fps)] );
    % Perform mean and median on 4th dimension (frames/times)
    mean_vid        = mean(im2double(one_sec_vid),4);
    median_vid      = median(im2double(one_sec_vid),4);

    figure
    subplot(1,2,1);


    imshow(mean_vid)
    title('Video Time Average')
    subplot(1,2,2);

    imshow(median_vid)
    title('Video Time Median')
end
