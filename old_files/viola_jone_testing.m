clc; close all; clear;

% Viola-Jones Based Facial Feature Detection

% List of image filenames
imageFiles = {'photo_5.jpg', 'photo_10.jpg', '1803191139-00001033.jpg', '1803191139-00000036.jpg'};

% Number of images
numImages = length(imageFiles);

% Create a Viola-Jones detector object for face detection
faceDetector = vision.CascadeObjectDetector();

% Create detectors for facial features (Eyes, Nose, Mouth)
eyeDetector = vision.CascadeObjectDetector('EyePairBig');
noseDetector = vision.CascadeObjectDetector('Nose');
mouthDetector = vision.CascadeObjectDetector('Mouth');

figure;

% Loop through each image, detect face and features, apply skin segmentation, and display
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
        faceRegion = imcrop(img, bboxFace);
        
        % Face region dimensions for mouth constraint
        faceHeight = size(faceRegion, 1);
        faceWidth = size(faceRegion, 2);
        
        % %% Eyes Detection with NMS
        % bboxEyes = step(eyeDetector, faceRegion);
        % if ~isempty(bboxEyes)
        %     % Apply non-maxima suppression to keep the largest eye pair
        %     areasEyes = bboxEyes(:, 3) .* bboxEyes(:, 4);
        %     [~, maxIndexEyes] = max(areasEyes);
        %     bboxEyes = bboxEyes(maxIndexEyes, :); % Keep the largest eyes pair
        % end
        
        %% Nose Detection with NMS
        bboxNose = step(noseDetector, faceRegion);
        if ~isempty(bboxNose)
            % Apply non-maxima suppression to keep the largest nose
            areasNose = bboxNose(:, 3) .* bboxNose(:, 4);
            [~, maxIndexNose] = max(areasNose);
            bboxNose = bboxNose(maxIndexNose, :); % Keep the largest nose
        end
        
        %% Mouth Detection with NMS and Location Constraint
        bboxMouth = step(mouthDetector, faceRegion);
        if ~isempty(bboxMouth)
            % Filter mouth detections to keep only those in the bottom half of the face
            bottomHalfMouths = bboxMouth(:, 2) > (faceHeight / 2);
            bboxMouth = bboxMouth(bottomHalfMouths, :);
            
            % Apply non-maxima suppression to keep the largest mouth in the bottom half
            if ~isempty(bboxMouth)
                areasMouth = bboxMouth(:, 3) .* bboxMouth(:, 4);
                [~, maxIndexMouth] = max(areasMouth);
                bboxMouth = bboxMouth(maxIndexMouth, :); % Keep the largest mouth
            end
        end
        
        %% Annotate Detected Features
        detectedFace = faceRegion; % Start with the cropped face region
        
        % % Annotate Eyes
        % if ~isempty(bboxEyes)
        %     detectedFace = insertShape(detectedFace, 'Rectangle', bboxEyes, 'Color', 'blue', 'LineWidth', 2);
        % end
        
        % Annotate Nose
        if ~isempty(bboxNose)
            detectedFace = insertShape(detectedFace, 'Rectangle', bboxNose, 'Color', 'green', 'LineWidth', 2);
        end
        
        % Annotate Mouth
        if ~isempty(bboxMouth)
            detectedFace = insertShape(detectedFace, 'Rectangle', bboxMouth, 'Color', 'red', 'LineWidth', 2);
        end
        
        %% Skin Segmentation
        
        % Convert the face region to HSV and RGB color spaces
        hsvFace = rgb2hsv(faceRegion);
        rgbFace = faceRegion;

        % Extract Red channel from RGB and Saturation from HSV
        redChannel = rgbFace(:,:,1);
        saturationChannel = hsvFace(:,:,1);

        % Create a mask from the facial features for skin region
        skinMask = false(size(faceRegion, 1), size(faceRegion, 2));
        % if ~isempty(bboxEyes)
        %     skinMask(bboxEyes(1,2):bboxEyes(1,2)+bboxEyes(1,4), bboxEyes(1,1):bboxEyes(1,1)+bboxEyes(1,3)) = true;
        % end
        if ~isempty(bboxNose)
            skinMask(bboxNose(1,2):bboxNose(1,2)+bboxNose(1,4), bboxNose(1,1):bboxNose(1,1)+bboxNose(1,3)) = true;
        end
        if ~isempty(bboxMouth)
            skinMask(bboxMouth(1,2):bboxMouth(1,2)+bboxMouth(1,4), bboxMouth(1,1):bboxMouth(1,1)+bboxMouth(1,3)) = true;
        end

        % Extract skin pixels using the mask
        skinRed = redChannel(skinMask);
        skinSaturation = saturationChannel(skinMask);
        
        % Create histograms of the skin pixels
        histRed = histcounts(skinRed, 256, 'Normalization', 'probability');
        histSaturation = histcounts(skinSaturation, 256, 'Normalization', 'probability');

        % Backproject the histograms onto the entire face region
        probabilityRed = histRed(double(redChannel)+1);  % Red channel backprojection

        % Convert the saturation values to indices in the range [1, 256]
        saturationIndices = round(saturationChannel * 255) + 1;
        
        % Ensure the indices are within the valid range [1, 256]
        saturationIndices = max(1, min(256, saturationIndices));
        
        % Backproject the histogram onto the saturation channel
        probabilitySaturation = histSaturation(saturationIndices);

        % Combine the probabilities
        skinProbability = probabilityRed .* probabilitySaturation;

        % Threshold the probability to segment the skin
        skinThreshold = 5E-5;
        skinSegment = skinProbability > skinThreshold;

        % Morphological Operations to Refine the Skin Segment
        % Define a structuring element. You can experiment with the size to get the best results.
        se = strel('disk', 5); % A disk-shaped structuring element with a radius of 5 pixels

        % Step 1: Apply Closing to connect small gaps in the skin mask
        skinSegment = imclose(skinSegment, se);

        % Step 2: Fill small holes within the detected skin regions
        skinSegment = imfill(skinSegment, 'holes');

        % Optional Step 3: Apply Opening to remove small noise pixels (small isolated white regions)
        skinSegment = imopen(skinSegment, se);

        %% Display the results
        
        % Display the face and features detection
        subplot(2, numImages, i);
        imshow(detectedFace);
        title(['Face & Features: Image ' num2str(i)]);
        
        % Display the skin segmentation with highlighted features
        subplot(2, numImages, numImages + i);
        imshow(skinSegment);
        title(['Skin Segmentation + Features: Image ' num2str(i)]);
        
    else
        % If no face detected, show the original image
        subplot(2, numImages, i);
        imshow(img);
        title(['No Face Detected: Image ' num2str(i)]);
        
        subplot(2, numImages, numImages + i);
        imshow(img);
        title(['No Skin Segmentation: Image ' num2str(i)]);  
    end
end
