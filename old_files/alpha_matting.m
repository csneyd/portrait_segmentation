clc; close all; clear;

% List of image filenames
imageFiles = {'photo_5.jpg', 'photo_10.jpg', '1803191139-00001033.jpg', '1803191139-00000036.jpg', '1803241220-00000379.jpg', '1803241220-00000455.jpg', '1803242348-00000223.jpg', '1803242348-00000303.jpg', '1803290249-00000352.jpg', '1803241944-00000092.jpg', '1803241944-00000180.jpg', '1803241944-00000241.jpg', '1803290249-00000142.jpg', '1803241944-00000549.jpg', '1803261700-00000010.jpg', '1803261700-00000240.jpg', '1803261700-00000348.jpg', '1803290249-00000131.jpg', '1803290100-00000210.jpg', '1803290100-00000343.jpg'};
% Number of images
numImages = length(imageFiles);

% Create a Viola-Jones detector object for face detection
faceDetector = vision.CascadeObjectDetector();

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
        sChannel = hsvImg(:,:,2);
        
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


        % Green Region Removal
        % Find pixels within the detected mask
        insideMaskIdx = find(combinedMask);
        
        % Get hue, saturation, and value for pixels inside the mask
        hMaskVals = hChannel(insideMaskIdx);
        sMaskVals = sChannel(insideMaskIdx);
        vMaskVals = vChannel(insideMaskIdx);
        
        % Define the HSV range for green pixels
        isGreen = (hMaskVals > 0.16 & hMaskVals < 0.45) & (vMaskVals > 0.16);
        
        % Create a binary mask for green regions
        greenMask = false(size(combinedMask));
        greenMask(insideMaskIdx(isGreen)) = true;
        
        % Morphological Processing on Green Regions
        se = strel('disk', 5); % Structuring element
        greenMask = imdilate(greenMask, se); % Expand green areas
        greenMask = imfill(greenMask, 'holes'); % Fill small holes
        greenMask = imerode(greenMask, se); % Refine by eroding excess dilation
        
        % Remove green regions from the hair mask
        combinedMask(greenMask) = false;


    
        % Morphological operations to refine the mask
        se = strel('disk', 6);
        combinedMask = imclose(combinedMask, se); % Close gaps
        combinedMask = imfill(combinedMask, 'holes'); % Fill small holes
    
        % Keep only the 3 largest connected components
        CC = bwconncomp(combinedMask); % Find connected components
        stats = regionprops(CC, 'Area'); % Get areas of components
        [~, sortedIdx] = sort([stats.Area], 'descend'); % Sort by area
        numComponentsToKeep = min(3, length(sortedIdx)); % Keep up to 3 components
        largestIdx = sortedIdx(1:numComponentsToKeep); % Indices of largest components
        
        % Create a new mask with only the largest components
        refinedMask = false(size(combinedMask));
        for j = 1:numComponentsToKeep
            refinedMask(CC.PixelIdxList{largestIdx(j)}) = true;
        end
        
        % Final combined mask
        combinedMask = refinedMask | largestRegionMask;


        %% Torso Segmentation
        
        % Step 1: Create a mask for pixels below the skin region
        [B_skin, ~] = bwboundaries(largestRegionMask, 'noholes');
        
        % Find the lowest Y-coordinate in largestRegionMask
        lowestY = 0;
        for k = 1:length(B_skin)
            boundary = B_skin{k}; % Get boundary points
            lowestY = max(lowestY, max(boundary(:, 1))); % Track the lowest Y value
        end
        
        % Create a mask covering everything below the lowest point of skin region
        belowSkinMask = false(size(img, 1), size(img, 2));
        belowSkinMask(lowestY:end, :) = true; % Only fill below this point

        
        % Compute the average hue from the skin mask
        skinHues = hChannel(largestRegionMask); 
        avgSkinHue = mean(skinHues(:)); 
        
        % Extract hue values from the belowSkinMask region
        hueValues = hChannel(belowSkinMask); 
        hueValues = hueValues(:); 
        
        % Find the two most common distinct hue values
        [counts, binCenters] = hist(hueValues, 50); 
        [sortedCounts, sortedIdx] = sort(counts, 'descend'); 
        topTwoHues = binCenters(sortedIdx(1:2)); 
        
        % Choose the hue farthest from the skin hue
        [~, maxDistIdx] = max(abs(topTwoHues - avgSkinHue)); 
        chosenHue = topTwoHues(maxDistIdx); 
        
        % Define threshold range around the chosen hue
        hueTolerance = 0.05; 
        lowerThreshold = max(0, chosenHue - hueTolerance);
        upperThreshold = min(1, chosenHue + hueTolerance);
        
        % Apply thresholding to segment the torso
        torsoMask = (hChannel >= lowerThreshold) & (hChannel <= upperThreshold);
        torsoMask = torsoMask & maskBelowMouth; 

        finalMask = torsoMask | combinedMask;
        

        %% Morphological Refinements
        
        se1 = strel('disk', 6); % Structuring element size (originally 5 for both)
        se2 = strel('disk', 7); % Structuring element size
        
        % Remove small noise (opening)
        finalMask = imopen(finalMask, se1);
        
        % Close small gaps (closing)
        finalMask = imclose(finalMask, se2);
        
        % Fill holes inside the torso mask
        finalMask = imfill(finalMask, 'holes');

        % Keep only the 2 largest connected components
        CC = bwconncomp(finalMask); % Find connected components
        stats = regionprops(CC, 'Area'); % Get areas of components
        [~, sortedIdx] = sort([stats.Area], 'descend'); % Sort by area
        numComponentsToKeep = min(2, length(sortedIdx)); % Keep up to 2 components
        largestIdx = sortedIdx(1:numComponentsToKeep); % Indices of largest components

        % Create a new mask with only the largest components
        refinedFinalMask = false(size(finalMask));
        for j = 1:numComponentsToKeep
            refinedFinalMask(CC.PixelIdxList{largestIdx(j)}) = true;
        end


        %% Apply mask

        % Apply mask to the original image
        maskedImg = img;
        for c = 1:size(img, 3)
            channel = maskedImg(:,:,c);
            channel(~refinedFinalMask) = 0; % Set pixels outside the mask to black
            maskedImg(:,:,c) = channel;
        end


        
        % Generate Trimap
        dilateSize = 10; % Controls the unknown region width
        se = strel('disk', dilateSize);
        
        sureFG = refinedFinalMask; % Your detected person mask
        sureBG = imdilate(~refinedFinalMask, se); % Expand background
        unknownRegion = ~(sureFG | sureBG); % Remaining pixels
        
        trimap = uint8(2 * sureFG + unknownRegion); % 0=BG, 1=Unknown, 2=FG
        
        % Convert Image to Double Precision
        imgDouble = im2double(img);
        
        % Apply Closed-Form Matting
        alphaMatte = closed_form_matting(imgDouble, trimap);
        
        % Extract Foreground
        foreground = imgDouble .* alphaMatte; % Soft segmentation
        



        %% Display the results

        % Display original image
        subplot(5, 8, subplotIdx);
        imshow(trimap, []), title('Trimap');
        subplotIdx = subplotIdx + 1;
        
        % Display corresponding mask
        subplot(5, 8, subplotIdx);
        subplot(1,3,3), imshow(foreground), title('Final Segmentation');
        subplotIdx = subplotIdx + 1;
        
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





function alpha = closed_form_matting(image, trimap, epsilon)
    % CLOSED_FORM_MATTING Implements the Closed-Form Matting by Levin et al.
    % image  : Input image (HxWx3)
    % trimap : Trimap (HxW) where:
    %          1 = foreground, 0 = background, 0.5 = unknown
    % epsilon: Regularization parameter (default: 1e-7)
    % alpha  : Output alpha matte (HxW)

    if nargin < 3
        epsilon = 1e-7;
    end

    image = im2double(image);
    [h, w, c] = size(image);
    n = h * w;

    % Convert image to column vectors
    imgVec = reshape(image, n, c);
    trimapVec = trimap(:);

    % Construct sparse Laplacian matrix
    [i_idx, j_idx, val] = compute_laplacian(image, epsilon);
    L = sparse(i_idx, j_idx, val, n, n);

    % Construct constraints matrix
    knownPixels = trimapVec ~= 0.5;
    D = spdiags(knownPixels, 0, n, n);

    % Solve for alpha using sparse linear system
    A = L + D;
    b = trimapVec;

    alphaVec = A \ b;
    alpha = reshape(alphaVec, h, w);
    
    % Clip values between 0 and 1
    alpha = max(0, min(1, alpha));
end






function [i_idx, j_idx, val] = compute_laplacian(image, epsilon)
    % Computes the Matting Laplacian for the input image
    [h, w, c] = size(image);
    n = h * w;

    % Convert image to grayscale if needed
    imgGray = rgb2gray(image);
    
    % Compute local mean and variance
    winSize = 3; % Window size
    halfWin = floor(winSize / 2);
    winN = winSize^2;

    % Image padding
    paddedImg = padarray(imgGray, [halfWin, halfWin], 'symmetric');
    
    % Compute mean and variance in a local window
    localMean = colfilt(paddedImg, [winSize winSize], 'sliding', @mean);
    localVar = colfilt(paddedImg, [winSize winSize], 'sliding', @var);
    
    % Compute the Laplacian entries
    i_idx = [];
    j_idx = [];
    val = [];
    
    for y = 1:h
        for x = 1:w
            idx = sub2ind([h, w], y, x);
            win = paddedImg(y:y+winSize-1, x:x+winSize-1);
            winVec = win(:);
            
            % Compute covariance matrix
            covMat = cov(winVec);
            I = eye(winN);
            L = I - (1 / (1 + trace(covMat) / winN + epsilon)) * (covMat + epsilon * I);
            
            % Assign values
            for i = 1:winN
                for j = 1:winN
                    i_idx = [i_idx; idx];
                    j_idx = [j_idx; idx];
                    val = [val; L(i, j)];
                end
            end
        end
    end
end
