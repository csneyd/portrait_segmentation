clc; close all; clear;

% List of image filenames
imageFiles = {'1803191139-00000004.jpg', '1803191139-00000114.jpg', '1803191139-00000116.jpg', '1803191139-00000214.jpg', '1803191139-00000268.jpg', '1803191139-00000276.jpg', '1803191139-00001039.jpg', '1803191139-00000316.jpg', '1803191139-00000345.jpg', '1803191139-00000391.jpg', '1803191139-00000410.jpg', '1803191139-00000525.jpg', '1803191139-00000528.jpg', '1803191139-00000596.jpg', '1803191139-00000598.jpg', '1803191139-00000646.jpg', '1803191139-00000784.jpg', '1803191139-00001037.jpg', '1803201916-00000171.jpg', '1803240740-00000042.jpg'};
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

        % Further crop into the face region by zooming in 84%
        zoomFactor = 0.84;  % Zooming factor
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
        
        % Define adaptive thresholds for the sat channel
        saturationMin = max(0, meanSat - 1.25 * stdSat); % Lower bound for sat
        saturationMax = min(1, meanSat + 1.25 * stdSat); % Upper bound for sat

        valChannelFace = hsvFace(:,:,3);

        meanVal = mean(valChannelFace(:));
        stdVal = std(valChannelFace(:));
        
        % Define adaptive thresholds for the val channel
        valueMin = max(0, meanVal - 1.5 * stdVal); % Lower bound for val
        valueMax = min(1, meanVal + 1.5 * stdVal); % Upper bound for val
        
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
        hairMask = hairMaskV & (hChannel < 0.2 | hChannel > 0.6);
        
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
        combinedMask = hairMask; % Start with the V-channel mask
        combinedMask(maskBelowMouth) = hairMask(maskBelowMouth) | hueEdges(maskBelowMouth);

    
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

        % Redefine thresholds
        saturationMin = max(0, meanSat - 1.75 * stdSat); % Lower bound for sat
        valueMax = 1;

        % Apply the existing skin thresholds only within the maskBelowMouth area
        hueSkinMask = (hChannel >= hueMin) & (hChannel <= hueMax) & maskBelowMouth;
        satSkinMask = (sChannel >= saturationMin) & (sChannel <= saturationMax) & maskBelowMouth;
        valSkinMask = (vChannel >= valueMin) & (vChannel <= valueMax) & maskBelowMouth;
        
        % Combine all three channels to get a refined skin mask
        skinColorMask = hueSkinMask & satSkinMask & valSkinMask;
        
        % Add skin-colored regions to the torso mask
        torsoMask = torsoMask | skinColorMask;
        % Combine with Torso Segmentation
        finalMask = torsoMask | combinedMask;


        %% Green Region Removal
        % Find pixels within the detected mask
        insideMaskIdx = find(finalMask);
        
        % Get hue, saturation, and value for pixels inside the mask
        hMaskVals = hChannel(insideMaskIdx);
        sMaskVals = sChannel(insideMaskIdx);
        vMaskVals = vChannel(insideMaskIdx);
        

        if chosenHue < 0.135 || chosenHue > 0.45
            % Define the HSV range for green pixels
            isGreen = (hMaskVals > 0.135 & hMaskVals < 0.45) & (vMaskVals > 0.16);
            
            % Create a binary mask for green regions
            greenMask = false(size(finalMask));
            greenMask(insideMaskIdx(isGreen)) = true;
            
            % Morphological Processing on Green Regions
            se = strel('disk', 5); % Structuring element
            greenMask = imdilate(greenMask, se); % Expand green areas
            greenMask = imfill(greenMask, 'holes'); % Fill small holes
            greenMask = imerode(greenMask, se); % Refine by eroding excess dilation
            
            % Remove green regions from the hair mask
            finalMask(greenMask) = false;
        end


        %% Edge Separation for Clean-Up
        
        % Compute Color Gradients in Lab Space
        labImg = rgb2lab(img); 
        L = labImg(:,:,1);
        a = labImg(:,:,2);
        b = labImg(:,:,3);
        
        % Compute gradient magnitude for each channel
        [GxL, GyL] = gradient(double(L));
        [GxA, GyA] = gradient(double(a));
        [GxB, GyB] = gradient(double(b));
        
        % Compute overall gradient magnitude
        gradMag = sqrt(GxL.^2 + GyL.^2 + GxA.^2 + GyA.^2 + GxB.^2 + GyB.^2);
        
        % Threshold to find strong edges
        edgeMask = gradMag > 12;  % Adjust threshold as needed
        
        % Keep only edges inside the detected mask
        edgeMask = edgeMask & finalMask; 
        
        % Break Connected Components at Detected Edges
        % Remove detected edges from the mask to break connected regions
        brokenMask = finalMask;
        brokenMask(maskAboveMouth) = finalMask(maskAboveMouth) & ~edgeMask(maskAboveMouth);
        
        % Perform morphological operations to clean up
        se = strel('disk', 3);  % Structuring element for smoothing
        brokenMask = imopen(brokenMask, se); % Remove noise
        brokenMask = imfill(brokenMask, 'holes'); % Fill small gaps
        
        % Label and Extract New Segments
        CC = bwconncomp(brokenMask); % Find new connected components
        % labeledMask = labelmatrix(CC); % Label segmented regions


        %% Morphological Refinements
        
        se1 = strel('disk', 10); % Structuring element size (originally 5 for both)
        se2 = strel('disk', 20); % Structuring element size
        
        % Remove small noise (opening)
        brokenMask = imopen(brokenMask, se1);
        
        % Close small gaps (closing)
        brokenMask = imclose(brokenMask, se2);
        
        % Fill holes inside the torso mask
        brokenMask = imfill(brokenMask, 'holes');


        %% Connect Head to Torso

        % Find the nth Lowest Row Containing Skin Pixels
        [rows, cols] = find(largestRegionMask); % Get all foreground pixel coordinates
        
        if isempty(rows)
            error('No skin pixels found in largestRegionMask.');
        end
        
        uniqueRows = unique(rows); % Get unique row indices
        numRows = length(uniqueRows);
        
        n = 15;
        if numRows >= n
            nthLowestRow = uniqueRows(end - (n - 1)); % nth lowest row
        else
            nthLowestRow = uniqueRows(1); % If fewer than n unique rows, take the lowest
        end
        
        % Fill All Columns from nthLowestRow to Bottom
        [~, width] = size(largestRegionMask);
        filledSkinMask = largestRegionMask; % Start with the existing mask
        
        for col = 1:width
            % Check if this column has a skin pixel in the nthLowestRow
            if largestRegionMask(nthLowestRow, col)
                % Fill everything below fifthLowestRow in this column
                filledSkinMask(nthLowestRow:end, col) = true;
            end
        end
        
        brokenMask = brokenMask | filledSkinMask; % Merge with any other masks if necessary


        %% Keep Largest Region and Fill Torso

        % Keep only the 2 largest connected components
        CC = bwconncomp(brokenMask); % Find connected components
        stats = regionprops(CC, 'Area'); % Get areas of components
        [~, sortedIdx] = sort([stats.Area], 'descend'); % Sort by area
        numComponentsToKeep = min(1, length(sortedIdx)); % Keep up to 2 components
        largestIdx = sortedIdx(1:numComponentsToKeep); % Indices of largest components

        % Create a new mask with only the largest components
        refinedFinalMask = false(size(brokenMask));
        for j = 1:numComponentsToKeep
            refinedFinalMask(CC.PixelIdxList{largestIdx(j)}) = true;
        end

        % Initialize the new filled mask
        filledTorsoMask = refinedFinalMask; % Start with the existing refined mask
        [height, width] = size(refinedFinalMask);
        
        % Scan and fill columns in maskBelowMouth
        for col = 1:width
            % Extract the column within the maskBelowMouth region
            columnMask = refinedFinalMask(:, col) & maskBelowMouth(:, col);
            
            % Find the first foreground pixel in this column
            rowIdx = find(columnMask, 1, 'first'); % Topmost foreground pixel
            
            % If a pixel was found, fill everything below it
            if ~isempty(rowIdx)
                filledTorsoMask(rowIdx:end, col) = true;
            end
        end

        
        %% Apply Human Template as Final Processing Step

        humanTemplateLow = im2gray(imread('silhouette3.png')); % Load a binary template
        humanTemplateLow = logical(imresize(humanTemplateLow, size(filledTorsoMask)));

        humanTemplateMedium = im2gray(imread('silhouette2.png')); % Load a binary template
        humanTemplateMedium = logical(imresize(humanTemplateMedium, size(filledTorsoMask)));

        humanTemplateHigh = im2gray(imread('silhigh.png')); % Load a binary template
        humanTemplateHigh = logical(imresize(humanTemplateHigh, size(filledTorsoMask)));

        if all(filledTorsoMask(1:200, :) == 0, 'all')  
            mask = filledTorsoMask & humanTemplateLow;   % For images with human lower in the image
        elseif all(filledTorsoMask(1:80, :) == 0, 'all')
            mask = filledTorsoMask & humanTemplateMedium; % For images with human in middle of image
        else 
            mask = filledTorsoMask & humanTemplateHigh; % For images with human near top of image
        end


        %% Apply mask

        % Apply mask to the original image
        maskedImg = img;
        for c = 1:size(img, 3)
            channel = maskedImg(:,:,c);
            channel(~mask) = 0; % Set pixels outside the mask to black
            maskedImg(:,:,c) = channel;
        end


        %% Display the results

        % Display original image
        subplot(5, 8, subplotIdx);
        imshow(img);
        title(['Original Image ' num2str(i)]);
        subplotIdx = subplotIdx + 1;
        
        % Display corresponding mask
        subplot(5, 8, subplotIdx);
        imshow(mask);
        title(['Mask Image ' num2str(i)]);
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
