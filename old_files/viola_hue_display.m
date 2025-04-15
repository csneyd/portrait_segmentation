clc; close all; clear;

% Viola-Jones Based Facial Feature Detection

% List of image filenames
imageFiles = {'photo_5.jpg', 'photo_10.jpg', '1803191139-00001033.jpg', '1803191139-00000036.jpg', '1803241220-00000379.jpg', '1803241220-00000455.jpg', '1803242348-00000223.jpg', '1803242348-00000303.jpg', '1803290249-00000352.jpg', '1803241944-00000092.jpg', '1803241944-00000180.jpg', '1803241944-00000241.jpg', '1803290249-00000142.jpg', '1803241944-00000549.jpg', '1803261700-00000010.jpg', '1803261700-00000240.jpg', '1803261700-00000348.jpg', '1803290249-00000131.jpg', '1803290100-00000210.jpg', '1803290100-00000343.jpg'};
% Number of images
numImages = length(imageFiles);

% Create a Viola-Jones detector object for face detection
faceDetector = vision.CascadeObjectDetector();

% Initialize a figure for 8x5 grid
figure;
set(gcf, 'Position', [100, 100, 1200, 800]); % Adjust figure size for better layout

% Counter to track the subplot index
subplotIdx = 1;

% Loop through each image, detect face, apply skin segmentation, and display
for i = 1:numImages
    % Read the current image
    img = imread(imageFiles{i});
    
    % Detect faces in the image
    bboxFace = step(faceDetector, img);  % Returns upper left corner and size of bounding box
    
    % Apply non-maxima suppression: keep only the largest face
    if ~isempty(bboxFace)
        % Compute the area of each bounding box and find the largest
        areas = bboxFace(:, 3) .* bboxFace(:, 4); % Width * Height
        [~, maxIndex] = max(areas);
        bboxFace = bboxFace(maxIndex, :); % Keep only the largest face
        
        % Crop the face region
        % faceRegion = imcrop(img, bboxFace);



        % Further crop into the face region by zooming in 75%
        zoomFactor = 0.84;  % Zooming factor (75% of the original size)
        centerX = bboxFace(1) + bboxFace(3) / 2;  % Center X coordinate of the bounding box
        centerY = bboxFace(2) + bboxFace(4) / 2;  % Center Y coordinate of the bounding box
        
        % Calculate new width and height
        newWidth = bboxFace(3) * zoomFactor;
        newHeight = bboxFace(4) * zoomFactor;
        
        % Calculate new top-left corner of the bounding box for cropping
        newX = centerX - newWidth / 2;
        newY = centerY - newHeight / 2;
        
        % Ensure the new bounding box is within image boundaries
        newX = max(1, newX);
        newY = max(1, newY);
        newWidth = min(newWidth, size(img, 2) - newX);
        newHeight = min(newHeight, size(img, 1) - newY);
        
        % Create the new bounding box for cropping
        newBbox = [newX, newY, newWidth, newHeight];
        
        % Crop the new face region
        faceRegion = imcrop(img, newBbox);




        
        %% Skin Segmentation Using Face Region Histograms
        
        % Convert the face region and original image to HSV and RGB color spaces
        hsvFace = rgb2hsv(faceRegion);
        hsvImg = rgb2hsv(img);
        rgbImg = img;
        
        % Extract Red channel from RGB and Hue from HSV (Face Region)
        redChannelFace = faceRegion(:,:,1);
        hueChannelFace = hsvFace(:,:,1);

        % Create histograms of the skin pixels (Face Region)
        histRed = histcounts(redChannelFace, 256, 'Normalization', 'probability');
        histHue = histcounts(hueChannelFace, 256, 'Normalization', 'probability');

        % Apply backprojection to the entire original image
        redChannelImg = rgbImg(:,:,1);
        hueChannelImg = hsvImg(:,:,1);

        % Backproject the histograms onto the entire original image
        probabilityRed = histRed(double(redChannelImg) + 1);  % Red channel backprojection

        % Convert the hue values of the entire image to indices in the range [1, 256]
        hueIndicesImg = round(hueChannelImg * 255) + 1;
        
        % Ensure the indices are within the valid range [1, 256]
        hueIndicesImg = max(1, min(256, hueIndicesImg));
        
        % Backproject the histogram onto the hue channel of the entire image
        probabilityHue = histHue(hueIndicesImg);

        % Combine the probabilities for the entire image
        skinProbability = probabilityHue .* probabilityRed;

        % Threshold the probability to segment the skin
        skinThreshold = 1E-4;  % Adjust this value as necessary
        skinSegment = skinProbability > skinThreshold;



        % Calculate the mean and standard deviation of the hue/sat/val values within the face region
        meanHue = mean(hueChannelFace(:));
        stdHue = std(hueChannelFace(:));
        
        % Define adaptive thresholds for the hue channel
        hueMin = max(0, meanHue - 0.75 * stdHue); % Lower bound for hue
        hueMax = min(1, meanHue + 0.75 * stdHue); % Upper bound for hue

        satChannelFace = hsvFace(:,:,2);

        meanSat = mean(satChannelFace(:));
        stdSat = std(satChannelFace(:));
        
        % Define adaptive thresholds for the hue channel
        saturationMin = max(0, meanSat - 1.25 * stdSat); % Lower bound for hue
        saturationMax = min(1, meanSat + 1.25 * stdSat); % Upper bound for hue

        valChannelFace = hsvFace(:,:,3);

        meanVal = mean(valChannelFace(:));
        stdVal = std(valChannelFace(:));
        
        % Define adaptive thresholds for the hue channel
        valueMin = max(0, meanVal - 1.5 * stdVal); % Lower bound for hue
        valueMax = min(1, meanVal + 1.5 * stdVal); % Upper bound for hue



        % Apply simple thresholding based on HSV values within the skinSegment mask
        % Define the acceptable range for Hue, Saturation, and Value channels
        % hueMin = 0.02; hueMax = 0.08;           % Adjust the range based on skin tone (e.g., [0, 0.1] for lighter tones)
        % saturationMin = 0.15; saturationMax = 0.8; % Adjust these ranges as necessary
        % valueMin = 0.20; valueMax = 1.0;         % Adjust these ranges as necessary
        
        % Extract the HSV channels of the original image
        hueChannel = hsvImg(:, :, 1);
        saturationChannel = hsvImg(:, :, 2);
        valueChannel = hsvImg(:, :, 3);
        
        % Apply the thresholds only within the skinSegment mask
        hueMask = (hueChannel >= hueMin) & (hueChannel <= hueMax) & skinSegment;
        saturationMask = (saturationChannel >= saturationMin) & (saturationChannel <= saturationMax) & skinSegment;
        valueMask = (valueChannel >= valueMin) & (valueChannel <= valueMax) & skinSegment;
        
        % Combine the masks with the original skinSegment mask
        thresholdedSkinMask = hueMask & saturationMask & valueMask;
        
        % Morphological Operations to Refine the Skin Segment (on thresholded mask)
        se = strel('disk', 9); % A disk-shaped structuring element with a radius of 9 pixels
        closedSkinMask = imclose(thresholdedSkinMask, se);
        filledSkinMask = imfill(closedSkinMask, 'holes');
        openedSkinMask = imopen(filledSkinMask, se);
        
        % Keep only the largest connected white region
        connectedComponents = bwconncomp(openedSkinMask);
        numPixels = cellfun(@numel, connectedComponents.PixelIdxList);
        [~, largestIdx] = max(numPixels);
        largestRegionMask = false(size(openedSkinMask));
        largestRegionMask(connectedComponents.PixelIdxList{largestIdx}) = true;

    %     %% Display the results
    % 
    %     % Display the face region
    %     subplot(3, numImages, i);
    %     imshow(img);
    %     title(['Detected Face: Image ' num2str(i)]);
    % 
    %     % Display the skin segmentation applied to the entire original image
    %     subplot(3, numImages, numImages + i);
    %     imshow(faceRegion);
    %     title(['Skin Segmentation: Image ' num2str(i)]);
    % 
    %     % Display the largest white region in the segmented mask
    %     subplot(3, numImages, 2 * numImages + i);
    %     imshow(largestRegionMask);
    %     title(['Largest Skin Region: Image ' num2str(i)]);
    % 
    % else
    %     % If no face detected, show the original image
    %     subplot(3, numImages, i);
    %     imshow(img);
    %     title(['No Face Detected: Image ' num2str(i)]);
    % 
    %     subplot(3, numImages, numImages + i);
    %     imshow(img);
    %     title(['No Skin Segmentation: Image ' num2str(i)]);  
    % 
    %     subplot(3, numImages, 2 * numImages + i);
    %     imshow(img);
    %     title(['No Largest Region: Image ' num2str(i)]); 


    % Display original image
    subplot(5, 8, subplotIdx);
    imshow(img);
    title(['Original Image ' num2str(i)]);
    subplotIdx = subplotIdx + 1;
    
    % Display corresponding mask
    subplot(5, 8, subplotIdx);
    imshow(largestRegionMask);
    title(['Mask Image ' num2str(i)]);
    subplotIdx = subplotIdx + 1;
    end
end
