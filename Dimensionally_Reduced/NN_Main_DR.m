clear all
close all

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tic 

sampling = 1;

addpath .\shared_Funcs\;
addpath .\NN_Implement\;
addpath .\images\;
ndim = 100;

% Load Dataset
[images, labels] = loadFaceImages('face_train.cdataset', sampling);

% Apply PCA, add ndim, or leave blank so the function will evaluate the 
% number needed to represent 95% total variance
[eigenVectors,eigenvalues,meanX,trainPCA] = PrincipalComponentAnalysis(images);

% Generate Model from Dataset
modelNN = NNTraining(trainPCA, labels);

% Plot Figure 
figure('NumberTitle', 'off', 'Name', 'NN Main DR'), hold on
colours= ['r.'; 'g.'; 'b.'; 'k.'; 'y.'; 'c.'; 'm.'; 'r+'; 'g+'; 'b+'; 'k+'; 'y+'; 'c+'; 'm+'];
count=0;

for i=min(labels):max(labels)
    count = count+1;
    indexes = find (labels == i);
    plot3(trainPCA(indexes,1),trainPCA(indexes,2),trainPCA(indexes,3),colours(count,:));
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% Load Dataset
[test_images, test_labels] = loadFaceImages('face_test.cdataset', sampling);
indexesNegatives = find (test_labels == -1);
indexesPositives = find (test_labels == 1);

test_images = [test_images(indexesNegatives, :); test_images(indexesPositives,:)];
test_labels = [test_labels(indexesNegatives); test_labels(indexesPositives)];

%For each testing image, we obtain a prediction based on our trained model
for i=1:size(test_images,1)
    
    testIm = test_images(i,:);
    testIm = (testIm - meanX) * eigenVectors;
    classificationResult(i,1) = NNTesting(testIm, modelNN);
    
end

%% Evaluation

    % Create the Confusion Matrix, Print all of the results
    createConfusionMatrix(test_labels, classificationResult);

toc
