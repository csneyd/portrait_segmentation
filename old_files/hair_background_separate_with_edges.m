clc; close all; clear;

% Load Viola-Jones detector for mouth detection
mouthDetector = vision.CascadeObjectDetector('Mouth', 'MergeThreshold', 16);

imgs = ["1803191139-00001033.jpg", "1803242348-00000303.jpg", ...
        "1803290249-00000142.jpg", "1803290100-00000210.jpg", ...
        '1803241220-00000455.jpg', '1803261700-00000348.jpg'];

for i = 1:length(imgs)
    img = imread(imgs(i));    

    subplot(2, length(imgs), i); imshow(img); title('Original Image');

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

    % Keep only the 3 largest connected components
    CC = bwconncomp(combinedMask);
    stats = regionprops(CC, 'Area'); % Get areas of components
    [~, sortedIdx] = sort([stats.Area], 'descend'); % Sort by area
    numComponentsToKeep = min(3, length(sortedIdx)); % Keep up to 3 components
    largestIdx = sortedIdx(1:numComponentsToKeep);
    
    % Create a new mask with only the largest components
    refinedMask = false(size(combinedMask));
    for j = 1:numComponentsToKeep
        refinedMask(CC.PixelIdxList{largestIdx(j)}) = true;
    end
    
    % Final refined mask
    combinedMask = refinedMask;

    % Step 1: Find the outline of the existing mask
    [B, ~] = bwboundaries(combinedMask, 'noholes'); % Get the outer boundary
    
    % Step 2: Extract the highest boundary points in each column
    height = size(combinedMask, 1);
    width = size(combinedMask, 2);
    topmostY = inf(1, width); % Stores the highest Y position for each column
    
    for k = 1:length(B)
        boundary = B{k}; % Get boundary points
        for idx = 1:size(boundary, 1)
            col = boundary(idx, 2);
            row = boundary(idx, 1);
            if row < topmostY(col)
                topmostY(col) = row; % Update if a new topmost pixel is found
            end
        end
    end
    
    % Step 3: Construct ordered X and Y coordinates
    validCols = find(topmostY < inf); % Columns with at least one mask pixel
    [sortedCols, sortIdx] = sort(validCols, 'ascend'); % Ensure increasing x-coordinates
    sortedRows = topmostY(sortedCols); % Get corresponding sorted Y values
    
    % Step 4: Append the bottom-most points to close the shape
    sortedCols = [sortedCols, fliplr(sortedCols)]; % Append the same x-coordinates in reverse
    sortedRows = [sortedRows, repmat(height, 1, length(sortedCols)/2)]; % Extend to bottom edge
    
    % Step 5: Generate the final filled mask
    filledTorsoMask = poly2mask(sortedCols, sortedRows, height, width);
    
    % Step 6: Overlay the traced outline on the original image
    outlineImage = img; % Copy original image
    for j = 1:length(validCols)
        outlineImage(sortedRows(j), sortedCols(j), :) = [255, 0, 0]; % Red line overlay
    end
    
    % Step 7: Combine with the original mask
    finalFilledMask = combinedMask;
    finalFilledMask(maskBelowMouth) = combinedMask(maskBelowMouth) | filledTorsoMask(maskBelowMouth);
    
    % Display results
    subplot(2, length(imgs), i + length(imgs)); imshow(finalFilledMask); title('Final Filled Mask');
    subplot(2, length(imgs), i); imshow(outlineImage); title('Traced Outline');

end
