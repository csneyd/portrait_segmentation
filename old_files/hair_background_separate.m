clc; close all; clear;

imgs = {'photo_5.jpg', 'photo_10.jpg', '1803191139-00001033.jpg', '1803191139-00000036.jpg', '1803241220-00000379.jpg', '1803241220-00000455.jpg', '1803242348-00000223.jpg', '1803242348-00000303.jpg', '1803290249-00000352.jpg', '1803241944-00000092.jpg', '1803241944-00000180.jpg', '1803241944-00000241.jpg', '1803290249-00000142.jpg', '1803241944-00000549.jpg', '1803261700-00000010.jpg', '1803261700-00000240.jpg', '1803261700-00000348.jpg', '1803290249-00000131.jpg', '1803290100-00000210.jpg', '1803290100-00000343.jpg'};

figure;
set(gcf, 'Position', [100, 100, 1200, 800]); % Adjust figure size for better layout
% Counter to track the subplot index
subplotIdx = 1;

for i = 1:length(imgs)
   img = imread(imgs{i});   

   % Convert to HSV color space
   hsvImg = rgb2hsv(img);
   vChannel = hsvImg(:,:,3); % Value (brightness)
  
   % Threshold for potential hair region
   hairMask = vChannel < 0.25; % Threshold for darker regions
  
   % Morphological operations
   se = strel('disk', 5);
   hairMask = imclose(hairMask, se); % Close gaps
   hairMask = imfill(hairMask, 'holes'); % Fill small holes
  
   % Remove small objects, keep connected regions close to the face
   hairMask = bwareaopen(hairMask, 100); %

   % Display original image
   subplot(5, 8, subplotIdx);
   imshow(img);
   title(['Original Image ' num2str(i)]);
   subplotIdx = subplotIdx + 1;
    
   % Display corresponding mask
   subplot(5, 8, subplotIdx);
   imshow(hairMask);
   title(['Mask Image ' num2str(i)]);
   subplotIdx = subplotIdx + 1;
  
end
