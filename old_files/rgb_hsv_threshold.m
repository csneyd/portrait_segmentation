clc; close all; clear;

% List of image filenames
% imageFiles = {'photo_5.jpg', 'photo_10.jpg', '1803191139-00001033.jpg', '1803191139-00000036.jpg'};
imageFiles = {'1803241220-00000379.jpg', '1803241220-00000455.jpg', '1803242348-00000223.jpg', '1803242348-00000303.jpg'};

% Parameters for hue thresholding (adjust these values as needed)
hueLower = 0.01; % Lower bound of hue for skin
hueUpper = 0.1;  % Upper bound of hue for skin

for i = 1:length(imageFiles)
    % Read the image
    img = imread(imageFiles{i});
    imgHSV = rgb2hsv(img);

    % Step 1: Apply Hue-based Thresholding
    hue = imgHSV(:,:,1);
    hueMask = (hue >= hueLower) & (hue <= hueUpper);

    % Step 2: Identify the Regions and Calculate RGB Thresholds
    maskedImg = img .* uint8(repmat(hueMask, [1, 1, 3]));
    
    % Get the pixels from the original RGB image that fall within the mask
    skinPixelsR = img(:,:,1);
    skinPixelsG = img(:,:,2);
    skinPixelsB = img(:,:,3);
    
    % Only keep values within the hue mask for calculating thresholds
    skinPixelsR = double(skinPixelsR(hueMask));
    skinPixelsG = double(skinPixelsG(hueMask));
    skinPixelsB = double(skinPixelsB(hueMask));

    % Calculate mean and standard deviation of each RGB channel
    meanR = mean(skinPixelsR); stdR = std(skinPixelsR);
    meanG = mean(skinPixelsG); stdG = std(skinPixelsG);
    meanB = mean(skinPixelsB); stdB = std(skinPixelsB);

    % Define RGB thresholds based on the mean ± standard deviation
    rMin = meanR - stdR; rMax = meanR + stdR;
    gMin = meanG - stdG; gMax = meanG + stdG;
    bMin = meanB - stdB; bMax = meanB + stdB;

    % Step 3: Apply Global RGB Thresholding
    redMask = (img(:,:,1) >= rMin) & (img(:,:,1) <= rMax);
    greenMask = (img(:,:,2) >= gMin) & (img(:,:,2) <= gMax);
    blueMask = (img(:,:,3) >= bMin) & (img(:,:,3) <= bMax);
    rgbMask = redMask & greenMask & blueMask;

    % Step 4: Final Skin Mask Combining Hue and RGB Thresholds
    finalSkinMask = hueMask & rgbMask;

    % Step 5: Morphological Operations to Refine the Mask
    % Fill holes in the mask
    refinedMask = imfill(finalSkinMask, 'holes');
    % Remove small objects (noise)
    refinedMask = bwareaopen(refinedMask, 300); % Minimum pixel area can be adjusted
    % Smooth edges using morphological closing
    se = strel('disk', 5); % Structuring element, can be adjusted
    refinedMask = imclose(refinedMask, se);

    % Step 6: Display the Results
    figure;
    subplot(1, 3, 1); imshow(img); title('Original Image');
    subplot(1, 3, 2); imshow(finalSkinMask); title('Before Morphological Operations');
    subplot(1, 3, 3); imshow(refinedMask); title('After Morphological Operations');

    % Optional: Apply final refined mask to the image for visualization
    segmentedSkin = img .* uint8(repmat(refinedMask, [1, 1, 3]));
    figure; imshow(segmentedSkin); title('Segmented Skin Region (Refined)');
end
