clc; close all; clear;

imgs = ["1803191139-00001033.jpg", "1803242348-00000303.jpg", ...
        "1803290249-00000142.jpg", "1803290100-00000210.jpg", ...
        "1803290249-00000352.jpg"];

% Parameters
hueThreshold = 0.11;
saturationThreshold = 0.4;
valueThreshold = 0.4;

% function for segmenting hair region
function hairMask = segmentHairRegion(image, seedPoints, hueThreshold, saturationThreshold, valueThreshold)
    % Convert image to HSV color space
    hsvImage = rgb2hsv(image);
    H = hsvImage(:, :, 1);  
    S = hsvImage(:, :, 2);  
    V = hsvImage(:, :, 3); 

    % Initialize mask and queue for region growing
    [rows, cols] = size(H);
    hairMask = false(rows, cols);
    queue = seedPoints;

    % Get seed points color information
    seedColors = cell(size(seedPoints, 1), 1);
    for i = 1:size(seedPoints, 1)
        seedColors{i} = [H(seedPoints(i, 2), seedPoints(i, 1)), ...
                         S(seedPoints(i, 2), seedPoints(i, 1)), ...
                         V(seedPoints(i, 2), seedPoints(i, 1))];
    end

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

            % Check against all seed colors
            isWithinThreshold = false;
            for i = 1:length(seedColors)
                seedHue = seedColors{i}(1);
                seedSaturation = seedColors{i}(2);
                seedValue = seedColors{i}(3);

                % Handle circular hue in distance
                hueDistance = min(abs(currentHue - seedHue), 1 - abs(currentHue - seedHue));

                % Check if color differences are within thresholds
                if hueDistance <= hueThreshold && ...
                   abs(currentSaturation - seedSaturation) <= saturationThreshold && ...
                   abs(currentValue - seedValue) <= valueThreshold
                    isWithinThreshold = true;
                    break;
                end
            end

            % If within threshold, add to mask and enqueue neighbors
            if isWithinThreshold
                hairMask(y, x) = true;

                % Enqueue neighboring pixels
                queue = [queue; x+1, y; x-1, y; x, y+1; x, y-1];
            end
        end
    end
end

% display results
figure;
numImages = length(imgs);

for i = 1:numImages
    % Read current image
    img = imread(imgs(i));
    
    % Detect face to determine seed points
    faceDetector = vision.CascadeObjectDetector;
    bbox = step(faceDetector, img);

    % Define two seed points for better coverage of hair variation
    seedPoint1 = [bbox(1) + round(bbox(3) / 5), bbox(2) - 40]; % Point above the left side of the face
    seedPoint2 = [bbox(1) + round(3 * bbox(3) / 8), bbox(2) - 60]; % Point above the right side of the face
    seedPoints = [seedPoint1; seedPoint2];

    % Segment hair region
    hairMask = segmentHairRegion(img, seedPoints, hueThreshold, saturationThreshold, valueThreshold);

    % Display results
    subplot(2, numImages, i);
    imshow(img);
    title(sprintf('Original Image %d', i));
    hold on;
    plot(seedPoints(:, 1), seedPoints(:, 2), 'r*', 'MarkerSize', 10);
    hold off;

    subplot(2, numImages, i + numImages);
    imshow(hairMask);
    title(sprintf('Segmented Hair %d', i));
end
