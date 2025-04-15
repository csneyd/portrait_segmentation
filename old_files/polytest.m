clc; close all; clear;    

xCoords = [100 15 70 40 61 10 126 120];
yCoords = [140 30 80 33 22 3 200 4];
height = 255;
width = 255;

filledTorsoMask = poly2mask(yCoords(:), xCoords(:), height, width);

figure;
imshow(filledTorsoMask);