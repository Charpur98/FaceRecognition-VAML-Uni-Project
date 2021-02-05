clear all
close all

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tic

sampling = 1;

addpath .\shared_Funcs\;
addpath .\KNN_Implement\;
addpath .\images\;

kVals = [1, 3, 5, 7, 9, 11];

% Load Dataset
[images, labels] = loadFaceImages('face_train.cdataset', sampling);

% Get HOG features for every image in Train dataset
trainHog = getHOG(images);

% Generate Model from Dataset
modelHogKNN = KNNTraining(trainHog, labels);

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Load Dataset
[test_images, test_labels] = loadFaceImages('face_test.cdataset', sampling);
indexesNegatives = find (test_labels == -1);
indexesPositives = find (test_labels == 1);

test_images = [test_images(indexesNegatives, :); test_images(indexesPositives,:)];
test_labels = [test_labels(indexesNegatives); test_labels(indexesPositives)];

% Get HOG features for every image in Test dataset
testHog = getHOG(test_images);

% For each testing image, obtain a prediction based on the trained model
for k=1:numel(kVals)
    
    for i=1:size(test_images,1)

        testIm= testHog(i,:);
        classificationResult(i,1) = KNNTesting(testIm, modelHogKNN, kVals(k));
        
    end
    
    fprintf('KNN with k=%d\n', kVals(k));
    
    % Create the Confusion Matrix, Print all of the results
    createConfusionMatrix(test_labels, classificationResult);
end

comparison = (test_labels == classificationResult);

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EVALUATION STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Plot ROC 
% Using inbuilt Matlab function to get ROC Curve
[X,Y] = perfcurve(test_labels, classificationResult, '1');

% Plotting ROC
plot(X, Y);
hold on
legend('Face','Location','SE');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC for KNN-Hog');

% We display the correctly classified images
figure('Name', 'Correct Classification', 'NumberTitle', 'off')
count = 0;
i = 1;
while (count<60)&&(i<=length(comparison))
   
    if comparison(i)
        count = count+1;
        subplot(10,10,count)
        Im = reshape(test_images(i,:),27,18);
        if(test_labels(i) == -1)
            imshow(uint8(Im)), title('Non Face');
        else
            imshow(uint8(Im)), title('Face');
        end
    end
    
    i = i+1;
    
end

% We display the incorrectly classified images
figure('Name', 'Wrong Classification', 'NumberTitle', 'off')
count = 0 ;
i = 1;
while (count<60)&&(i<=length(comparison))
    
    if ~comparison(i)
        count = count+1;
        subplot(10,10,count)
        Im = reshape(test_images(i,:),27,18);
        if(test_labels(i) == -1)
            imshow(uint8(Im)), title('Non Face');
        else
            imshow(uint8(Im)), title('Face');
        end
    end
    
    i=i+1;
end

toc
