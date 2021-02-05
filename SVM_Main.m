clear all
close all

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tic

sampling = 1;

addpath .\shared_Funcs\;
addpath .\SVM_Implement\;
addpath .\SVM_Implement\SVM-KM\;
addpath .\images\;

% Load Dataset
[images, labels] = loadFaceImages('face_train.cdataset', sampling);

% Generate Model from Dataset
modelSVM = SVMTraining(images, labels);

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Loading testing labels and images
[test_images, test_labels] = loadFaceImages('face_test.cdataset', sampling);

for i=1:size(test_images,1)
    
    testIm = test_images(i,:);
    classificationResult(i,1) = SVMTesting(testIm, modelSVM);

end


% Create the Confusion Matrix, Print all of the results
createConfusionMatrix(test_labels, classificationResult);

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
title('ROC for SVM-Main');

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

save SVM_MAIN_MODEL modelSVM
