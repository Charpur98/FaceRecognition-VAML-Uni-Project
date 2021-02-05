clear all
close all

% Importing functions
addpath .\shared_Funcs\;
addpath .\SVM_Implement\;
addpath .\SVM_Implement\SVM-KM;
addpath .\images\;
addpath .\Detectors\;
addpath .\Models\;

% Load both of the previously generated Models
load('SVM_GABOR_MODEL.mat'); % Model name is modelGaborSVM
load('SVM_HOG_MODEL.mat'); % Model name is modelHogSVM
windowSize = [18, 27]; % Window Size is the same as the images from dataset

images{1} = imread('images/im1.jpg');
images{2} = imread('images/im2.jpg');
images{3} = imread('images/im3.jpg');
images{4} = imread('images/im4.jpg');

prefix = "results/hog_gabor_SVM_No_Pre_Process";
suffix = ".jpg";

tic % Start Timer for Entire Detection Process

for iNumber = 1:4
    fprintf("Starting image number %d\n", iNumber);
    figure(iNumber);
    % Convert Image to a Matrix 
    thisImg = cell2mat(images(iNumber));
    
    % Contrast Enhancement performed for first 3 images. Skips Image 4
    % because results are better without it
    if iNumber ~= 4
        thisImg = adapthisteq(thisImg);
    end
    
    imshow(thisImg); % Display Image

    % Get the Bounding Boxes using the Gabor Detector
    bBoxes = SVM_Hog_Gabor_Detector(modelHogSVM, modelGaborSVM, thisImg, windowSize);
    fprintf("\nBounding Boxes got for image number %d\n", iNumber);
        
    % Perform Non Maximum Suppression to reduce amount of Bounding Boxes
    bBoxes = NonMaxSuppression(bBoxes, 0.1);
    
    % Iterate through Bounding Boxes and impose them onto the image
    for i = 1:size(bBoxes,1) 
        rectangle('Position',[bBoxes(i, 1),bBoxes(i, 2),bBoxes(i, 3) - bBoxes(i, 1),bBoxes(i, 4) - bBoxes(i, 2)],'LineWidth',3, 'EdgeColor','b');
    end
    fprintf("Bounding Boxes drawn for image number %d\n", iNumber);
    
    filename = strcat(prefix, int2str(iNumber), suffix);
    set (gcf, 'PaperPositionMode', 'manual','PaperPosition',[0, 0, 50, 30])
    print(figure(iNumber),filename,'-djpeg'); 
    fprintf("Saved image number %d\n", iNumber);
    fprintf("Finished image number %d\n", iNumber);
end

toc % End Timer for Entire Detection Process