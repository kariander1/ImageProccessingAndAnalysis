
%% sec. 1B


% Load video:
vid_path        = '../Corsica.mp4';
video           = VideoReader(vid_path);
fps             = get(video,'FrameRate');
% selected second = 4[min]X60[sec] + 10 [sec] = 250
% this should produce 250 frames (=10 seconds of video)
vid_scene     = read(video, [ (fps*250 +1)  (fps*250 +10*fps)] );
implay(vid_scene,fps);


num_of_frames = size(vid_scene,4);
% calculate new rows count to be 2/3 of original count
new_row_count = round(size(vid_scene,1)*(2/3));

% create empty video with new dimensions
gray_corsica = zeros([new_row_count size(vid_scene,2) num_of_frames] , class(vid_scene));

for k=1 :num_of_frames
    % iterate over each frame, convert it to grayscale and resize it to 2/3
    % of lower image
    rgb_frame = rgb2gray(vid_scene(:,:,:,k));
    cut_image = rgb_frame(round(size(vid_scene,1)*(1/3))+1 : size(vid_scene,1),:);
    gray_corsica(:,:,k) = cut_image;
end   

%% sec. 1C
% Create empty panorama frame
panorama_frame = uint8(zeros([new_row_count size(vid_scene,2)*2.5]));

%% sec. 1D
frame_padding = 10;
% Pick reference frame to be the middle one in scene
ref_frame = gray_corsica(:,:,round(num_of_frames/2));
ref_frame = ref_frame(:,frame_padding:end-frame_padding);
% Prepare to concatenate 3 matrices horizontaly
% Create appendix 1 to be the zero panorama frame, this appendix will end
% where the ref image needs to be inserted, so we take the center of the
% columns count and substract half of the ref image width to figure out
% where to end the first appendix.

ref_frame_offset = round(size(panorama_frame,2)/2)-round(size(ref_frame,2)/2);
appendix_1 = uint8(panorama_frame(:,1:ref_frame_offset));
% Appendix 2 will be added after the ref image, so we take the center of the
% columns count and add half of the ref image width.
appendix_2 = uint8(panorama_frame(:,round(size(panorama_frame,2)/2)+round(size(ref_frame,2)/2):end));
% Concatenate all componets
panorama_frame_with_ref = uint8(horzcat(appendix_1 ,ref_frame ,appendix_2));

% Show new panorama with ref image
figure
subplot(2,1,1);
imshow(ref_frame,[]);
title('Reference Image')

subplot(2,1,2);
imshow(panorama_frame_with_ref,[]);
title('Panoramic with Reference')



%% sec. 1E
% Choose delta in # frames to take prev and next frames
frame_delta = 100;
% Extract later frame
late_frame = gray_corsica(:,:,round(num_of_frames/2)+frame_delta);
% Extract earlier frame
earlier_frame = gray_corsica(:,:,round(num_of_frames/2)-frame_delta);

% Remove vertical black framing
late_frame = late_frame(:,frame_padding:end - frame_padding);
earlier_frame = earlier_frame(:,frame_padding:end - frame_padding);

% Display chosen frames
figure
subplot(1,2,1);
imshow(earlier_frame,[]);
title('Earlier Frame')

subplot(1,2,2);
imshow(late_frame,[]);
title('Later Frame')


%% sec. 1F

% Select cropping for earlier frame (road+fences)
earlier_region_start    = 10; %Cropped image start pixel
earlier_region_end      = 400;%Cropped image end pixel 
earlier_width= earlier_region_end - earlier_region_start; %Calc cropping width
% Extract cropped frame:
earlier_region = earlier_frame(:,earlier_region_start:earlier_region_end);

% Select cropping for later frame (bush+rocks)
later_region_start = 280; %Cropped image start pixel
later_region_end   = size(late_frame,2)-10; %Cropped image end pixel 
later_width= later_region_end - later_region_start; %Calc cropping width
% Extract cropped frame:
later_region = late_frame(1:end,later_region_start:later_region_end);

% Apply correlation of selected cropping with ref_image:
[ealier_row,ealier_col] = matchCorr(earlier_region,ref_frame);
[later_row,later_col]   = matchCorr(later_region,ref_frame);

% Display ealier frame with selected cropping
figure
subplot(3,2,2);
imshow(earlier_frame,[]);
rectangle('Position',[earlier_region_start,0,earlier_region_end-earlier_region_start,size(earlier_region,1)],'LineWidth',3,'LineStyle','-','EdgeColor','b')
title('Earlier Frame corr')

% Display later frame with selected cropping
subplot(3,2,1);
imshow(late_frame,[]);
rectangle('Position',[later_region_start,0,later_region_end-later_region_start,size(earlier_region,1)],'LineWidth',3,'LineStyle','-','EdgeColor','r')
title('Later Frame corr')

% Display later frame crop with coordinates of the correlation
subplot(3,2,4);
imshow(earlier_region,[]);

title(strcat('Earlier Frame [row=',int2str(ealier_row),',column=',int2str(ealier_col),']'));


% Display earlier frame crop with coordinates of the correlation
subplot(3,2,3);
imshow(later_region,[]);
title(strcat('Later Frame [row=',int2str(later_row),',column=',int2str(later_col),']'))

% Display reference frame and display the cropping boundries within the
% ref_image and show center of max correlation
subplot(3,2,[5 6]);
imshow(ref_frame,[]);
title('Ref image regions');
rectangle('Position',[later_col-5,later_row-5,10,10],'LineWidth',3,'LineStyle','-','EdgeColor','r')
rectangle('Position',[later_col-later_width/2,0,later_width,240],'LineWidth',3,'LineStyle','-','EdgeColor','r')
rectangle('Position',[ealier_col-5,ealier_row-5,10,10],'LineWidth',3,'LineStyle','-','EdgeColor','b')
rectangle('Position',[ealier_col-earlier_width/2,0,earlier_width,240],'LineWidth',3,'LineStyle','-','EdgeColor','b')
%% sec. 1G
% Calculate offsets of panorama and new frames in respect to it
later_col_to_ref = later_col+ref_frame_offset+1;
earlier_col_to_ref = ealier_col+ref_frame_offset+1;
panorama_late_left_offset = later_col_to_ref - round(later_region_start +later_width/2 );
panorama_early_left_offset = earlier_col_to_ref - round(earlier_region_start +earlier_width/2 );

% Create clean panorama image with later frame
panorama_with_later = panorama_frame;
panorama_with_later(:,panorama_late_left_offset+1:(panorama_late_left_offset+size(late_frame,2)))=late_frame;
% Create clean panorama image with earlier frame
panorama_with_earlier = panorama_frame;
panorama_with_earlier(:,panorama_early_left_offset+1:(panorama_early_left_offset+size(earlier_frame,2)))=earlier_frame;

% Create empty panorama image
panorama_end = panorama_frame;

% Vector of matrices to iterate over
matrices = {panorama_with_earlier panorama_with_later panorama_frame_with_ref};

for i = 1:size(panorama_end,1)    
    for j = 1:size(panorama_end,2)
        % Foreach pixel in panorama image, calculate average pixel value.
        pixel_count = 0;
        pixel_sum   = 0;
        % Iterate over images of the panoramas (earlier,reference,later)
        for m= 1:3
             if (matrices{m}(i,j) ~= 0)
                 % If there is a valid pixel accumulate value and image
                 % count
                 pixel_count=pixel_count+1;
                 pixel_sum=pixel_sum+matrices{m}(i,j);
             end
        end
        % Calculate the average value of the pixel
        panorama_end(i,j) = round(pixel_sum/pixel_count);           
    end
end

% Show the final panorama image:
figure
imshow(panorama_end,[]);
title("Panorama Final (WOW!)")
%% sec. 2A

% Load and show keyboard
figure
keyboard_path     = '..\keyboard.jpg';
im_keyboard      = imread(keyboard_path);
[keyboard_width ,keyboard_height] = size(im_keyboard);
imshow(im_keyboard,[]);
title('Keyboard')

%% sec. 2B
% Create structural elements as requested
SE_horizontal = strel('line',8,0);
SE_vertical = strel('line',8,90);
% Erode image horizontally and vertically
erode_horizontal = imerode(im_keyboard,SE_horizontal);
erode_vertical = imerode(im_keyboard,SE_vertical);

% Show eroded image
figure
subplot(2,2,[1 2]);
imshow(im_keyboard,[]);
title('Keyboard original')
rectangle('Position',[keyboard_height/2,keyboard_width/2-4,1,8],'LineWidth',3,'LineStyle','-','EdgeColor','r')
subplot(2,2,3);
imshow(erode_horizontal,[]);
title("Horizontal Erosion")

subplot(2,2,4);
imshow(erode_vertical,[]);
title("Vertical Erosion")

%% sec. 2C
% Sum the eroded images and display
erode_sum = erode_horizontal+erode_vertical;
figure

subplot(2,2,[1 2]);
imshow(erode_sum,[]);
title("Erosion Sum")

% Get binary representation of image
binary_sum = im2bw(erode_sum,0.2);
subplot(2,2,[3 4]);
imshow(binary_sum,[]);
title("Erosion Binary")

%% sec. 2D

% Perform and display negation and filter with 8X8 median filter
nega_keyboard = imcomplement(binary_sum);
nega_keyboard_filered = medfilt2(nega_keyboard,[8 8]);


subplot(2,2,[1 2]);
imshow(nega_keyboard,[]);
title("Negative Keyboard")

subplot(2,2,[3 4]);
imshow(nega_keyboard_filered,[]);
title("Negative Filtered")

%% sec. 2E
% Create structural element of a square 8X8
% Erode and display the negated keyboard
SE_square = strel('square',8);
square_erosion = imerode(nega_keyboard_filered,SE_square);
imshow(square_erosion,[]);
title("Erosion Square");

%% sec. 2F
% Sharpen and display the eroded & negated keyboard
square_erosion = uint8(square_erosion);
prod_keyboard = imsharpen(square_erosion .* im_keyboard);

imshow(prod_keyboard,[]);
title("Sharpen");

%% sec. 2G
% Apply threshold and display image
prod_keyboard_threshold = (prod_keyboard>=100);
imshow(prod_keyboard_threshold,[]);
title("Keyboard Final");

%% sec. 3A

% Load video:
vid_path        = '../Flash Gordon Trailer.mp4';
video           = VideoReader(vid_path);
fps             = get(video,'FrameRate');
implay(vid_path);
% Perform analysis on two frames:
DenoiseFrame(510,video);
DenoiseFrame(960,video);

function DenoiseFrame(frame_num,video)

% Extract & display red channel of selected frame
selected_frame     = read(video, frame_num );
imshow(selected_frame,[]);
title(strcat('Selected frame ',frame_num));
selected_frame_red = imresize(selected_frame(:,:,1),0.5);

imshow(selected_frame_red,[]);

% Add electron noise
a = 3;
electron_im = uint16(double(selected_frame_red)*a);
% Add poisson distributed noise
noisy_im_rounded = round(imnoise(electron_im,'poisson')/a);
% Clip the noise so values won't exceed [0,255] range
noisy_im_clipped = noisy_im_rounded;
noisy_im_clipped(noisy_im_clipped>255)=255;
noisy_im_clipped(noisy_im_clipped<0)=0;
noisy_im_clipped=double(noisy_im_clipped);
% Display noisy image
figure;
subplot(1,2,1)
imshow(selected_frame_red,[]);
title("Image with out noise");


subplot(1,2,2)
imshow(noisy_im_clipped,[]);
title("Image with noise");



%% sec. 3C

% Perform denoise by L2 method, and calculate error to original image
[Xout, Err1, Err2] = DenoiseByL2(noisy_im_clipped, selected_frame_red, 50, 0.5);

% Display semilog graph of noises
figure
semilogy(Err1,'Color','r');
hold on;
semilogy(Err2,'Color','b');
legend('Err1','Err2');
title("Denoise By L2 error");

%% sec. 3D
% Perform denoise by TV method, and calculate error to original image
[Xout, Err1, Err2] = DenoiseByTV(noisy_im_clipped, selected_frame_red, 200, 20);

% Display semilog graph of noises
figure
semilogy(Err1,'Color','r');
hold on;
semilogy(Err2,'Color','b');
legend('Err1','Err2');
title("Denoise By TV error");
end
function [Xout, Err1, Err2]=DenoiseByTV(Y, X, numIter, lambda)
% Create column stack of images
X_cs =reshape(X,[],1);
Y_cs =reshape(Y,[],1);
% Define epsilon
epsilon = 1e-3;
% Define u_k
u_k = 100 *epsilon;
X_k = Y_cs; % X_0 image initial value
% Create error vectors
Err1 = zeros([1 numIter]);
Err2 = zeros([1 numIter]);

for iterator = 1:numIter 
    % Calculate X/Y gradients of image
    [X_k_grad_x,X_k_grad_y] = imgradientxy(reshape(X_k,size(X)));   
    % Calculate magnitude of gradients
    X_k_abs = X_k_grad_x.^2 +X_k_grad_y.^2;

    % Convert gradients to column stacks
    X_k_grad_x_cs = reshape(X_k_grad_x,[],1);
    X_k_grad_y_cs = reshape(X_k_grad_y,[],1);

    % Calculate the total variation value
    TV = sum(sqrt( X_k_grad_x_cs.^2 +X_k_grad_y_cs.^2));
    
    % Calculate the denomniator of the divergence expression
    normalization_term = sqrt( X_k_abs +epsilon^2);
    % Calculate divergence
    D = divergence( X_k_grad_x./normalization_term, X_k_grad_y./normalization_term);
    
    % Perform step to next X_k
    X_k = X_k +(u_k/2)*(2*(Y_cs-X_k)+reshape(lambda*D,[],1));
    
    
    
    % Calculate erors
    Err1(iterator) = (X_k-Y_cs).'*(X_k-Y_cs)+lambda*(TV);
    Err2(iterator) = (X_k-double(X_cs)).'*(X_k-double(X_cs));
    
    % Print every 50th iteration the image
   if (mod(iterator,50) == 0)
      figure;
       imshow(reshape(X_k,size(X)),[]);
       title(strcat('Iteration ',int2str(iterator)));
   end
end
% Return the final image after reshaping to original dimesions
Xout=reshape(X_k,size(X));
end
%% sec. 3B


function [Xout, Err1, Err2] = DenoiseByL2(Y, X, numIter, lambda)

% Create column stack of images
X_cs =reshape(X,[],1);
Y_cs =reshape(Y,[],1);
% Create kernel matrix
D_kernel = ( [0 1 0;
              1 -4 1;
              0 1 0]);
X_k = Y_cs; % X_0 initial value
% Calculate expression for G_k
identity_plus_lambda = (eye(size(D_kernel)))+lambda*conv2((D_kernel.'),D_kernel,'same');
% Create empty errors vector
Err1 = zeros([1 numIter]);
Err2 = zeros([1 numIter]);
 
for iterator = 1:numIter    
   
    % Calc G_k
    G_k = reshape(imfilter(reshape(X_k,size(X)), identity_plus_lambda ),[],1)-Y_cs;
    
    % Calc u_k
    u_k = ((G_k.')*G_k)/(G_k.'*reshape(imfilter(reshape(G_k,size(X)),identity_plus_lambda),[],1));
    
    % Step X_k
    X_k = X_k-u_k*G_k;
    
    % Calculate errors
    Err1_conv = reshape(imfilter(reshape(X_k,size(X)),D_kernel),[],1); % Intermediate value
    Err1(iterator) = (X_k-Y_cs).'*(X_k-Y_cs)+lambda*(Err1_conv.')*(Err1_conv);
    Err2(iterator) = (X_k-double(X_cs)).'*(X_k-double(X_cs));

    % Display every 10th image
   if (mod(iterator,10) == 0)
      figure;
       imshow(reshape(X_k,size(X)),[]);
       title(strcat('Iteration ',int2str(iterator)));
   end
end
% Return the final image after reshaping to original dimesions
Xout=reshape(X_k,size(X));


end
%% sec. 1A
function [row,column] = matchCorr(Obj, Im)
    
    % Rotate 180 degrees to make conv into correlation
    rotated_obj = rot90(rot90(Obj));
    % Find maximum value of correlation of the obj with itself
    maxObj      = max(conv2(Obj,rotated_obj,'same'),[],'all');
    % Get correlation image
    correlation = conv2(Im,rotated_obj,'same');
    
    % Extract linear index of min diffrence between correlation and
    % expected value (objmax)
    abs_diff_matrix =abs(correlation-maxObj);
    max_corr     = min(abs_diff_matrix,[],'all');
    [row,column] = find(abs_diff_matrix==max_corr);
    

end
