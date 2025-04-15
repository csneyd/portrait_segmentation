clc; close all; clear;

% img = imread('1803191139-00001033.jpg');
% img = imread('1803261700-00000348.jpg');
% img = imread('1803241220-00000455.jpg');
% img = imread('1803290249-00000142.jpg');
% img = imread('1803241944-00000241.jpg');
img = imread('1803242036-00000059.jpg');
figure('color','white'); subplot(2, 4, 1); imshow(img); title('Original Image');

grayImg = rgb2gray(img);

% Viola Jones for initial region capture
faceDetector = vision.CascadeObjectDetector;
bbox = step(faceDetector, img);
if ~isempty(bbox)
    % Calculate new dimensions for the head area
    newHeight = bbox(4) * 2;  % 2 times the height
    newWidth = bbox(3) * (5/3);  % 5/3 times the width
    
    % Calculate new top-left corner to keep the box centered
    newX = bbox(1) + (bbox(3) - newWidth) / 2;
    newY = bbox(2) + (bbox(4) - newHeight) / 2;
    
    % Define the new head area
    headArea = [newX, newY, newWidth, newHeight];
    % headArea = [bbox(1), bbox(2), bbox(3), bbox(4)]; % Extend head area
end

% Crop head area
headCrop = imcrop(img, headArea);
grayHeadCrop = im2gray(headCrop);


[M, N] = size(grayImg);                   % Dims of input img

P = 2 * M;                                  % Define row padding
Q = 2 * N;                                  % Define column padding

f = padarray(grayImg, [M N], 0, 'post');  % Pad image with zeros
F = fft2(double(f));                        % Perform FFT
Fshift = fftshift(F);                       % Shift DC to center

D_0 = [5, 10, 15, 20, 30, 50, 60];               % Array of 5 cutoff frequencies

% Generate filtered images for 5 different frequencies
for i=1:length(D_0)
    H = hairFilter(P, Q, D_0(i), 25);            % Generate Gaussian LP filter
    G = Fshift .* H;                        % Apply filter to input img freq spectrum
    
    g = ifft2(fftshift(G));                 % Shift back and return to spatial domain
    g_o = real(g);                   % Take real part
    freqMap = g_o(1:M, 1:N);                    % Cut out portion of image that is not zeros


    meanFreq = mean(freqMap(:));  % Calculate the mean of the frequency map
    stdFreq = std(freqMap(:));    % Calculate the standard deviation of the frequency map
    
    % Define threshold: mean minus standard deviation
    threshold = meanFreq - stdFreq;
    
    % Apply thresholding
    freqMask = freqMap <= threshold;
    
    % Convert logical map to binary
    freqMask = double(freqMask);


    se = strel('disk', 5); % A disk-shaped structuring element with a radius of 5 pixels
    closedMap = imclose(freqMask, se);
    openedMap = imopen(closedMap, se);

    subplot(2, 4, i+1); imshow(openedMap, []); title(['Thresholded Freq Map, D_0 = ', num2str(D_0(i))]);

end


% Color Analysis
ycbcrImg_head = rgb2ycbcr(headCrop);
ycbcrImg = rgb2ycbcr(img);
Cb_head = ycbcrImg_head(:,:,2);
Cr_head = ycbcrImg_head(:,:,3);
Cb = ycbcrImg(:,:,2);
Cr = ycbcrImg(:,:,3);

% Define hair color model using top region of head area
sampleCb = Cb_head(round(size(headCrop,1)*0.1):round(size(headCrop,1)*0.4), :);
sampleCr = Cr_head(round(size(headCrop,1)*0.1):round(size(headCrop,1)*0.4), :);

figure; imshow(headCrop);
figure; imshow(sampleCb);
figure; imshow(sampleCr);

meanCb = mean(sampleCb(:));
stdCb = std(double(sampleCb(:)));
meanCr = mean(sampleCr(:));
stdCr = std(double(sampleCr(:)));

% Threshold Cb and Cr to create color mask
colorMask = (Cb > meanCb - stdCb) & (Cb < meanCb + stdCb) & ...
            (Cr > meanCr - stdCr) & (Cr < meanCr + stdCr);

se = strel('disk', 5); % A disk-shaped structuring element with a radius of 5 pixels

openedCMap = imopen(colorMask, se);
closedCMap = imclose(openedCMap, se);
figure; imshow(closedCMap);

% Fusion
fusionMask = freqMask | openedCMap;


% Display Results
overlay = img;
overlay(:,:,1) = overlay(:,:,1) + uint8(255 * fusionMask);
figure; imshow(overlay); title('Hair Segmentation Result');