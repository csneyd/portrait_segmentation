clc; close all; clear;

% List of image filenames
imageFiles = {'photo_5.jpg', 'photo_10.jpg', '1803191139-00001033.jpg', '1803191139-00000036.jpg'};

% Define hue and saturation range for skin tone segmentation
hueMin = 0.02;  % Approx 0 degrees (slightly above to avoid pure red)
hueMax = 0.08;  % Approx 50 degrees (end of typical skin tone range)
satMin = 0.15;   % Minimum saturation for skin tone (to avoid very pale areas)
satMax = 0.8;   % Maximum saturation for skin tone (to avoid oversaturated areas)

% Loop over each image file
for i = 1:length(imageFiles)
    % Read image
    img = imread(imageFiles{i});
    
    % Convert image to HSV color space
    hsvImg = rgb2hsv(img);
    
    % Extract hue and saturation channels
    hueChannel = hsvImg(:,:,1);
    satChannel = hsvImg(:,:,2);
    
    % Plot histograms of hue and saturation values
    figure;
    subplot(2,3,1);
    histogram(hueChannel(:), 'BinWidth', 0.01, 'Normalization', 'probability');
    title(['Hue Histogram for ', imageFiles{i}]);
    xlabel('Hue Value');
    ylabel('Frequency');
    
    subplot(2,3,2);
    histogram(satChannel(:), 'BinWidth', 0.01, 'Normalization', 'probability');
    title(['Saturation Histogram for ', imageFiles{i}]);
    xlabel('Saturation Value');
    ylabel('Frequency');
    
    % Segment skin pixels based on hue and saturation range
    skinMask = (hueChannel >= hueMin) & (hueChannel <= hueMax) & ...
               (satChannel >= satMin) & (satChannel <= satMax);
           
    % Display the binary skin mask
    subplot(2,3,3);
    imshow(skinMask);
    title('Binary Skin Mask');
    
    % Apply mask to get segmented skin region in the original image
    skinSegmented = img;
    skinSegmented(repmat(~skinMask, [1, 1, 3])) = 0; % Set non-skin pixels to black
    
    % Display original, skin mask, and segmented images
    subplot(2,3,4);
    imshow(img);
    title('Original Image');
    
    subplot(2,3,5);
    imshow(skinSegmented);
    title('Segmented Skin Region');
    
    % Overlay binary skin mask on original image for visual comparison
    overlayImg = img;
    overlayImg(:,:,1) = overlayImg(:,:,1) + uint8(skinMask) * 128;
    subplot(2,3,6);
    imshow(overlayImg);
    title('Overlay of Skin Mask on Original');
end
