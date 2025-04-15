clc; close all; clear;

% List of image filenames
imgs = ["1803191139-00001033.jpg", "1803242348-00000303.jpg", ...
        "1803290249-00000142.jpg", "1803290100-00000210.jpg", ...
        "1803290249-00000352.jpg"];

% Parameters
hueThreshold = 0.1;
saturationThreshold = 0.3;
valueThreshold = 0.3;

% Define the hair segmentation function
function hairMask = segmentHairRegion(image, seedPoint, hueThreshold, saturationThreshold, valueThreshold)
    % Convert image to HSV color space
    hsvImage = rgb2hsv(image);
    H = hsvImage(:, :, 1); 
    S = hsvImage(:, :, 2);  
    V = hsvImage(:, :, 3);  
    
    % Initialize mask and queue for region growing
    [rows, cols] = size(H);
    hairMask = false(rows, cols);
    queue = seedPoint;
    
    % Get seed point color information
    seedHue = H(seedPoint(2), seedPoint(1));
    seedSaturation = S(seedPoint(2), seedPoint(1));
    seedValue = V(seedPoint(2), seedPoint(1));
    
    % Perform region growing
    while ~isempty(queue)
        % Dequeue a point
        point = queue(1, :);
        queue(1, :) = [];
        
        x = point(1);
        y = point(2);
        
        % Check if point is within bounds and not already part of the mask
        if x > 0 && y > 0 && x <= cols && y <= rows && ~hairMask(y, x)
            currentHue = H(y, x);
            currentSaturation = S(y, x);
            currentValue = V(y, x);
            
            % Handle circular hue distance
            hueDistance = min(abs(currentHue - seedHue), 1 - abs(currentHue - seedHue));
            
            % Check if color differences are within thresholds
            if hueDistance <= hueThreshold && ...
               abs(currentSaturation - seedSaturation) <= saturationThreshold && ...
               abs(currentValue - seedValue) <= valueThreshold
                
                hairMask(y, x) = true;
                
                % Enqueue neighboring pixels
                queue = [queue; x+1, y; x-1, y; x, y+1; x, y-1];
            end
        end
    end
end

% Prepare figure for subplots
numImages = length(imgs);
figure;

for i = 1:numImages
    % Read the current image
    img = imread(imgs(i));
    
    % Detect face to determine the seed point
    faceDetector = vision.CascadeObjectDetector;
    bbox = step(faceDetector, img);
    
    % Calculate the dynamic seed point based on the face bounding box
    x_center = bbox(1) + bbox(3) / 2; % Horizontal center of the face
    y_top = bbox(2) - bbox(4) * 0.2; % A fraction above the top of the bounding box
    
    % Ensure the seed point is within image bounds
    seedPoint = [max(1, round(x_center)), max(1, round(y_top))];
    
    % Perform hair segmentation
    hairMask = segmentHairRegion(img, seedPoint, hueThreshold, saturationThreshold, valueThreshold);
    
    % Display the original image and segmented mask
    subplot(2, numImages, i); 
    imshow(img); 
    hold on;
    plot(seedPoint(1), seedPoint(2), 'r*', 'MarkerSize', 10); % Mark the seed point
    hold off;
    title(sprintf('Original %d', i));
    
    subplot(2, numImages, numImages + i);
    imshow(hairMask); 
    title(sprintf('Hair Mask %d', i));
end
