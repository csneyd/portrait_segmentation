clc; close all; clear;

% List of image filenames
imageFiles = {'1803241220-00000379.jpg', '1803241220-00000455.jpg', '1803242348-00000223.jpg', '1803242348-00000303.jpg', "1803191139-00001033.jpg", "1803290249-00000142.jpg", "1803290100-00000210.jpg", '1803261700-00000348.jpg'};

% Number of images
numImages = length(imageFiles);

% Create a Viola-Jones detector object for face detection
faceDetector = vision.CascadeObjectDetector();

figure;

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
        se = strel('disk', 9); % A disk-shaped structuring element with a radius of 5 pixels
        closedSkinMask = imclose(thresholdedSkinMask, se);
        filledSkinMask = imfill(closedSkinMask, 'holes');
        openedSkinMask = imopen(filledSkinMask, se);
        
        % Keep only the largest connected white region
        connectedComponents = bwconncomp(openedSkinMask);
        numPixels = cellfun(@numel, connectedComponents.PixelIdxList);
        [~, largestIdx] = max(numPixels);
        largestRegionMask = false(size(openedSkinMask));
        largestRegionMask(connectedComponents.PixelIdxList{largestIdx}) = true;




        %% Hair Segmentation

        % Load Viola-Jones detector for mouth detection
        mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 16);

        % Convert to HSV color space
        hsvImg = rgb2hsv(img);
        vChannel = hsvImg(:,:,3); % Value (brightness)
        hChannel = hsvImg(:,:,1); % Hue
        
        % Threshold for potential hair region using Value channel
        hairMaskV = vChannel < 0.25; % Threshold for darker regions
        hairMaskV = hairMaskV & (hChannel < 0.2 | hChannel > 0.6);
        
        % Detect mouth region using Viola-Jones
        bbox = step(mouthDetector, img); % Detect mouth bounding box
        if ~isempty(bbox)
            % Assuming the largest bounding box corresponds to the mouth
            [~, idx] = max(bbox(:,3) .* bbox(:,4));
            mouthBox = bbox(idx, :);
        else
            % If no mouth is detected, assume the whole image is for processing
            mouthBox = [1, size(img, 1) / 2, size(img, 2), size(img, 1) / 2];
        end
        
        % Define regions: above and below the mouth
        maskAboveMouth = true(size(img, 1), size(img, 2));
        maskAboveMouth(mouthBox(2) + mouthBox(4):end, :) = false;
        maskBelowMouth = ~maskAboveMouth;
    
        % Compute gradient of the Hue channel for the whole image
        [Gx, Gy] = gradient(double(hChannel)); % Gradient in x and y directions
        gradientMag = sqrt(Gx.^2 + Gy.^2); % Magnitude of the gradient
    
        % Threshold the gradient to find regions with rapid hue changes
        hueEdges = gradientMag > 0.11; % Adjust threshold as needed
    
        % Combine masks based on regions
        combinedMask = hairMaskV; % Start with the V-channel mask
        combinedMask(maskBelowMouth) = hairMaskV(maskBelowMouth) | hueEdges(maskBelowMouth);
    
        % Morphological operations to refine the mask
        se = strel('disk', 6);
        combinedMask = imclose(combinedMask, se); % Close gaps
        combinedMask = imfill(combinedMask, 'holes'); % Fill small holes
    
        % Keep only the 5 largest connected components
        CC = bwconncomp(combinedMask); % Find connected components
        stats = regionprops(CC, 'Area'); % Get areas of components
        [~, sortedIdx] = sort([stats.Area], 'descend'); % Sort by area
        numComponentsToKeep = min(3, length(sortedIdx)); % Keep up to 5 components
        largestIdx = sortedIdx(1:numComponentsToKeep); % Indices of largest components
        
        % Create a new mask with only the largest components
        refinedMask = false(size(combinedMask));
        for j = 1:numComponentsToKeep
            refinedMask(CC.PixelIdxList{largestIdx(j)}) = true;
        end
        
        % Final combined mask
        combinedMask = refinedMask | largestRegionMask;

        % Apply mask to the original image
        maskedImg = img;
        for c = 1:size(img, 3)
            channel = maskedImg(:,:,c);
            channel(~combinedMask) = 0; % Set pixels outside the mask to black
            maskedImg(:,:,c) = channel;
        end




        %% Display the results
        
        % Display the face region
        subplot(4, numImages, i);
        imshow(img);
        title(['Detected Face: Image ' num2str(i)]);
        
        % Display the skin segmentation applied to the entire original image
        subplot(4, numImages, numImages + i);
        imshow(faceRegion);
        title(['Skin Segmentation: Image ' num2str(i)]);
        
        % Display the largest white region in the segmented mask
        subplot(4, numImages, 2 * numImages + i);
        imshow(combinedMask);
        title(['Skin and Hair Segmentation: Image ' num2str(i)]);

        % Display the segmented region
        subplot(4, numImages, 3 * numImages + i);
        imshow(maskedImg);
        title(['Segmented Region: Image ' num2str(i)]);
        
    else
        % If no face detected, show the original image
        subplot(3, numImages, i);
        imshow(img);
        title(['No Face Detected: Image ' num2str(i)]);
        
        subplot(3, numImages, numImages + i);
        imshow(img);
        title(['No Skin Segmentation: Image ' num2str(i)]);  

        subplot(3, numImages, 2 * numImages + i);
        imshow(img);
        title(['No Largest Region: Image ' num2str(i)]); 
    end
end
