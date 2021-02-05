clear all
close all

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tic

sampling = 1;

addpath .\shared_Funcs\;
addpath .\SVM_Implement\;
addpath .\SVM_Implement\SVM-KM\;
addpath .\images\;
addpath .\ImageProcessing\;

[images, labels] = loadFaceImages('face_train.cdataset', sampling);

% Train HOG
trainHog = getHOG(images);

% Apply PCA, add ndim, or leave blank so the function will evaluate the 
% number needed to represent 95% total variance
[eigenVectors,eigenvalues,meanX,trainPCA] = PrincipalComponentAnalysis(trainHog);

% Perform training
modelSVM = SVMTraining(trainPCA, labels);

% Plot Figure 
figure('NumberTitle', 'off', 'Name', 'SVM HOG DR'), hold on
colours= ['r.'; 'g.'; 'b.'; 'k.'; 'y.'; 'c.'; 'm.'; 'r+'; 'g+'; 'b+'; 'k+'; 'y+'; 'c+'; 'm+'];
count=0;

for i=min(labels):max(labels)
    count = count+1;
    indexes = find (labels == i);
    plot3(trainPCA(indexes,1),trainPCA(indexes,2),trainPCA(indexes,3),colours(count,:));
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Loading testing labels and images
[test_images, test_labels] = loadFaceImages('face_test.cdataset', sampling);

indexesNegatives = find (test_labels == -1);
indexesPositives = find (test_labels == 1);

test_images = [test_images(indexesNegatives, :); test_images(indexesPositives,:)];
test_labels = [test_labels(indexesNegatives); test_labels(indexesPositives)];

testHog = getHOG(test_images);

for i=1:size(test_images,1)
    
    testIm = testHog(i,:);
    testIm = (testIm - meanX) * eigenVectors;
    classificationResult(i,1) = SVMTesting(testIm, modelSVM);

end


% Create the Confusion Matrix, Print all of the results
createConfusionMatrix(test_labels, classificationResult);

comparison = (test_labels == classificationResult);

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ EVALUATION STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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