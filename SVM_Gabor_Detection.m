clear all 
close all

% Importing functions
addpath .\shared_Funcs\;
addpath .\SVM_Implement\;
addpath .\SVM_Implement\SVM-KM;
addpath .\images\;
addpath .\Detectors\;

% Load the model that was generated from SVM_Gabor script
load('Models/SVM_GABOR_MODEL.mat');
windowDims = [18, 27];

images{1} = imread('images/im1.jpg');
images{2} = imread('images/im2.jpg');
images{3} = imread('images/im3.jpg');
images{4} = imread('images/im4.jpg');

% Create Prefix and Suffix for saving resulting images into results
% directory
prefix = "results/gabor_SVM_im";
suffix = ".jpg";

for imNum = 1:4
    fprintf("Processing Image %d\n", imNum);
    figure(imNum);
    % Convert Image to a Matrix 
    curIm = cell2mat(images(imNum));
    
    % Contrast Enhancement using built in Matlab funtion
    curIm = adapthisteq(curIm)
    
    imshow(curIm)
    
    % Get the Bounding Boxes using the Gabor Detector
    boundingBoxes = SVM_Gabor_Detector(modelGaborSVM, curIm, windowDims);
    fprintf("Collected Bounding Boxes for Image No. %d\n", imNum);
    
    % Perform Non Maximum Suppression
    boundingBoxes = NonMaxSuppression(boundingBoxes, 0.1);
    
    % Iterate through Bounding Boxes and impose them onto the image
    for i = 1:size(boundingBoxes, 1)
        rectangle('Position',[boundingBoxes(i, 1),boundingBoxes(i, 2),boundingBoxes(i, 3) - boundingBoxes(i, 1),boundingBoxes(i, 4) - boundingBoxes(i, 2)],'LineWidth',3, 'EdgeColor','g');
    end
    
    fprintf("Bounding Boxes drawn for image number %d\n", imNum);
    
    filename = strcat(prefix, int2str(imNum), suffix);
    set (gcf, 'PaperPositionMode', 'manual','PaperPosition',[0, 0, 50, 30])
    fprintf("Saved image number %d\n", imNum);
    
    fprintf("Finished image number %d\n", imNum);
end