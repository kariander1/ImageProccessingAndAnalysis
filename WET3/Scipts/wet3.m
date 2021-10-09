%% sec. 1A


% Load video:
vid_path        = '../Time - Pink Floyd.mp4';
video           = VideoReader(vid_path);
fps             = get(video,'FrameRate');
% selected second = 4[min]X60[sec] + 10 [sec] = 250
% this should produce 250 frames (=10 seconds of video)

implay(vid_path);

%% sec. 1B

selected_frame_num_sec = 33 ;
selected_frame_num = int16(fps*(selected_frame_num_sec-1)+1);
% Sample the selected frame with respect to the FPS:
selected_frame     = read(video, selected_frame_num );


%% sec. 1C

% Make a copy of the frame:
selected_frame_copy = selected_frame;

% Display selected frame with rectancle surronding row 300
figure
imshow(selected_frame_copy,[]);
title('Selected Frame with highlight');
rectangle('Position',[300,295,670,20],'LineWidth',2,'LineStyle','-','EdgeColor','r')

% Sample grayscale of the red channel
selected_frame_red = selected_frame(:,:,1);
selected_row = selected_frame_red(300,:);

% Display grayscale levels as a function of column
figure
plot(selected_row);
xlabel('Col #') 
ylabel('Grayscale level') 
title("Grayscale as a function of column");

%% sec. 1D
delta_x = 64;
[rows,cols] = size(selected_frame_red);
center_col = cols/2;

sampled_image = selected_frame_red(:,delta_x:delta_x:end);
cols_sampled = size(sampled_image,2);
figure

imshow(selected_frame_copy,[])
for iterator = 1:cols_sampled 
    rectangle('Position',[iterator*delta_x-1,0,2,rows],'LineWidth',1,'LineStyle','-','EdgeColor','r')
end
title("Sampled columns");
figure
imshow(sampled_image,[]);
title("Sampled Image");

%% sec. 1E

resized_image = imresize(sampled_image ,[rows,cols] );
figure
imshow(resized_image,[]);
title("Resized Image");

%% sec. 1F

figure
imshow(resized_image,[]);

title('Resized Frame with highlight');
rectangle('Position',[300,295,670,20],'LineWidth',2,'LineStyle','-','EdgeColor','r')

selected_row = resized_image(300,:);

% Display grayscale levels as a function of column
figure
plot(selected_row);
xlabel('Col #') 
ylabel('Grayscale level') 
title("Grayscale Resized as a function of column");

%% sec. 1G

% Define start and end second
start_sec = 30;
end_sec = 45;

% Extract start and end frames
start_frame =int16((fps*(start_sec) +1));
end_frame =int16(fps*end_sec);

% Load frames and display video
vid_scene     = read(video, [ (start_frame)  (end_frame)] );
implay(vid_scene,fps);

%% sec. 1H

% Sample video every 16th frame
delta_p = 16;
sampled_vid_scene = vid_scene(:,:,:,1:delta_p:end);

%% sec. 1I

% Define new video
new_vid_name = 'new_vid.mp4';
new_vid = VideoWriter(new_vid_name,'MPEG-4');

% Interpolate on new video with ZOH on 16 frames
open(new_vid);
for k = 1:size(sampled_vid_scene,4)
   for p=1:delta_p
     
       writeVideo(new_vid,sampled_vid_scene(:,:,:,k));
   end
end

% Close video and play it
close(new_vid);
implay(new_vid_name);


%% sec 2B

% Define empty matrix
data_mat = zeros(4096,13233);
path_to_images ='../LFW/' ;
file_extension = '*.pgm';

% Load all images from dir:
images = dir(strcat(path_to_images, file_extension));
for i=1:numel(images)
    image_filename = images(i).name;
    image = imread(strcat(path_to_images, image_filename));
    % Perform column stack of image
    image_cs =reshape(image,[],1);
    % Concatenate image to matrix
    data_mat(:,i) = image_cs;
end

%% sec 2C
 % Create covariance matrix
cov_mat = cov(data_mat.');

%% sec 2D + H
% Project images with 10 base vectors
projectImages(10,cov_mat,data_mat);

% Project images with 570 base vectors
projectImages(570,cov_mat,data_mat);

%% sec 3B

im_downey              = rgb2gray(im2double(imread('..\Downey.jpg')));
im_ironman              = rgb2gray(im2double(imread('..\ironman.jpg')));

n=4;

% Generate pyramid for downey
[G_n_downey,L_n_downey] = pyrGen(im_downey,n);
figure
for i=1:n+1
    
    subplot(2,n+1,i);
    imshow(G_n_downey{i},[]);
    title(strcat("Gaussian ",int2str(i-1)));
    subplot(2,n+1,i+n+1);
    
    imshow(L_n_downey{i},[]);
    title(strcat("Lapalcian ",int2str(i-1)));
end

% Generate pyramid for ironman
[G_n_ironman,L_n_ironman] = pyrGen(im_ironman,n);
figure
for i=1:n+1
    subplot(2,n+1,i);    
    imshow(G_n_ironman{i},[]);
    title(strcat("Gaussian ",int2str(i-1)));
    subplot(2,n+1,i+n+1);    
    imshow(L_n_ironman{i},[]);
    title(strcat("Lapalcian ",int2str(i-1)));
end
%% sec 3D

% Call reconstruction function for images
downey_recon = imRecon(L_n_downey);
ironman_recon = imRecon(L_n_ironman);

% Calc diff images and display them
downey_diff = im_downey - downey_recon;
ironman_diff = im_ironman - ironman_recon;
figure
imshowpair(downey_diff,downey_recon,'montage');
err_downey = immse(im_downey,downey_recon);
title(strcat("Reconstructed images [diff image , reconstructed image] , MSE = ",num2str(err_downey)));

figure
imshowpair(ironman_diff,ironman_recon,'montage');
err_ironman = immse(im_ironman,ironman_recon);
title(strcat("Reconstructed images [diff image , reconstructed image] , MSE = ",num2str(err_ironman)));

%% sec 2D
function projectImages(k,cov_mat,data_mat)
    
    % Get k eigen vectors and values and plot them
    [eigen_vectors ,eigen_vals_mat]  = eigs(cov_mat,k);
    eigen_vals = diag(eigen_vals_mat);
    figure

    plot(eigen_vals);
    title(strcat(int2str(k)," largest eigenvalues as of K"));
    xlabel("k [#]");
    ylabel("value");


    %% sec 2E

    figure
    % Show 4 most significant vectors
    for i=1:4
        subplot(2,2,i)
        image_mat = reshape(eigen_vectors(:,i),[64,64]);
        imshow(image_mat,[]);
        title(strcat(int2str(i)," most significant"));
    end

    %% sec 2F
    % Create projection matrix
    p = (eigen_vectors.') * data_mat;

    %% sec 2G


    lord_hutton_index = 8140;
    lleyton_hewitt_index = 8122;
    madonna_index = 8304;
    trump_index =3138;

    selected_indices = [lord_hutton_index ,lleyton_hewitt_index,madonna_index, trump_index];


    figure
    
    % Foreach image calc MSE and display reconstruction vs original
    for i=1:4

        subplot(2,2,i)
        selected_index = selected_indices(i);
        selected_image = reshape(data_mat(:,selected_index),[64,64]);
        projected_image = reshape(eigen_vectors*p(:,selected_index),[64,64]);
        % Calc MSE
        err = immse(selected_image,projected_image);
        imshowpair(selected_image,projected_image,'montage');
        title(strcat('MSE = ',int2str(err)));
    end
end
%% 3A
% Li=Gi-Expand(G(i+1))
function [G_n,L_n] = pyrGen(img_grayscale,n)
    % Wrapper function for pyramid generation
    [G_n,L_n] = pyrGenRec(img_grayscale,n,0);
end
function [G_n,L_n] = pyrGenRec(img_grayscale,n,i)

    % Stop condition when reaching last step
    if i==n
       G_n{i+1} = img_grayscale;
       L_n{i+1} = img_grayscale;
       return
    end
    
    % Calc next gaussian image and current laplacian image
    G_next_level =impyramid(imgaussfilt(img_grayscale),'reduce');
    L_current = img_grayscale - impyramid(G_next_level,'expand');
    
    % Call recursion
    [G_n,L_n] = pyrGenRec(G_next_level,n,i+1);
    
    % Concatenate to given cell:
    G_n{i+1} = img_grayscale;
    L_n{i+1} = L_current;
end

%% 3C
function rec_image = imRecon(L_n)

    % Get first image
    rec_image =L_n{1};
    
    % Foreach image, expand it relevant times and add it
    for i=2:size(L_n,2)
        rec_image = rec_image+ expandImage(L_n{i},i-1);
    end
end
function image =expandImage(image,n)
    % Expand image n times
    for i=1:n
        image = impyramid(image,'expand');
    end
end