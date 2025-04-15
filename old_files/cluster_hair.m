clc; close all; clear;

%% Hair Segmentation Using Viola-Jones and Clustering
% Load the input images
imageFiles = {'1803191139-00001033.jpg', '1803242348-00000303.jpg', ...
              '1803290249-00000142.jpg', '1803290100-00000210.jpg', ...
              '1803241220-00000455.jpg', '1803261700-00000348.jpg'};

figure;
for imgIdx = 1:length(imageFiles)
    inputImage = imread(imageFiles{imgIdx});

    % Convert to grayscale for face detection
    grayImage = rgb2gray(inputImage);

    % Detect the head region using Viola-Jones
    faceDetector = vision.CascadeObjectDetector;
    faceBBox = step(faceDetector, grayImage);

    if isempty(faceBBox)
        warning('No face detected in image: %s.', imageFiles{imgIdx});
        continue;
    end

    % Expand the bounding box to capture more of the hair
    expansionFactor = 1.75; % Increase the size by 75%
    expandedBBox = faceBBox;
    expandedBBox(1) = max(1, faceBBox(1) - (expansionFactor - 1) * faceBBox(3) / 2);
    expandedBBox(2) = max(1, faceBBox(2) - (expansionFactor - 1) * faceBBox(4) / 2);
    expandedBBox(3) = min(size(inputImage, 2) - expandedBBox(1), faceBBox(3) * expansionFactor);
    expandedBBox(4) = min(size(inputImage, 1) - expandedBBox(2), faceBBox(4) * expansionFactor);

    % Extract the expanded head region
    headRegion = imcrop(inputImage, expandedBBox);

    % Convert the entire image to HSV color space for clustering
    hsvFullImage = rgb2hsv(inputImage);
    hChannel = hsvFullImage(:, :, 1);
    sChannel = hsvFullImage(:, :, 2);
    vChannel = hsvFullImage(:, :, 3);

    % Extract the HSV values from the head region for clustering
    hsvHeadRegion = rgb2hsv(headRegion);
    hHead = hsvHeadRegion(:, :, 1);
    sHead = hsvHeadRegion(:, :, 2);
    vHead = hsvHeadRegion(:, :, 3);
    hsvPixels = double([hHead(:), sHead(:), vHead(:)]);

    % Perform K-means clustering with 3 clusters
    k = 3;
    [clusterIdx, ~] = kmeans(hsvPixels, k, 'Distance', 'sqeuclidean', 'Replicates', 5);

    % Reshape cluster labels to image dimensions of the head region
    pixelLabels = reshape(clusterIdx, size(hHead));

    % Determine which cluster corresponds to hair (assume darker tone is hair)
    meanIntensity = zeros(k, 1);
    for i = 1:k
        meanIntensity(i) = mean(vHead(pixelLabels == i));
    end
    [~, hairCluster] = min(meanIntensity);

    % Calculate mean and standard deviation of the hair cluster
    hairPixels = hsvPixels(clusterIdx == hairCluster, :);
    hairMean = mean(hairPixels);
    hairStd = std(hairPixels);

    % Create a binary mask for the hair region based on mean plus/minus std across the entire image
    hairMask = (abs(hChannel - hairMean(1)) <= hairStd(1)) & ...
               (abs(sChannel - hairMean(2)) <= hairStd(2)) & ...
               (abs(vChannel - hairMean(3)) <= hairStd(3));

    % Apply morphological operations to smooth the hair mask
    hairMask = imclose(hairMask, strel('disk', 12));
    hairMask = imfill(hairMask, 'holes'); % Fill holes


    % Get the hue values of pixels within the hair mask
    hueValuesInHairMask = hChannel(hairMask);
    
    % Find the 20th percentile of hue values
    hueThreshold = prctile(hueValuesInHairMask, 20);
    
    % Refine the hair mask to retain only pixels with hue values <= the threshold
    refinedHairMask = hairMask & (hChannel <= hueThreshold);
    
    % Apply morphological operations to smooth the refined hair mask
    refinedHairMask = imclose(refinedHairMask, strel('disk', 12));
    refinedHairMask = imfill(refinedHairMask, 'holes'); % Fill holes
    
    % Apply the refined mask to the input image to segment the hair
    segmentedHair = inputImage;
    segmentedHair(repmat(~refinedHairMask, [1, 1, 3])) = 0;


    % Display results
    subplot(length(imageFiles), 3, 3 * imgIdx - 2); imshow(inputImage); title(sprintf('Original Image %d', imgIdx));
    subplot(length(imageFiles), 3, 3 * imgIdx - 1); imshow(refinedHairMask); title(sprintf('Refined Hair Mask %d', imgIdx));
    subplot(length(imageFiles), 3, 3 * imgIdx); imshow(segmentedHair); title(sprintf('Segmented Hair %d', imgIdx));
end
