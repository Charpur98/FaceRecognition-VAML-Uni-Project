clear all
close all

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tic

sampling = 1;

addpath .\shared_Funcs\;
addpath .\KNN_Implement\;
addpath .\images\;
ndim = 100;

% Load Dataset
[images, labels] = loadFaceImages('face_train.cdataset', sampling);

% Apply PCA, add ndim, or leave blank so the function will evaluate the 
% number needed to represent 95% total variance
[eigenVectors,eigenvalues,meanX,trainPCA] = PrincipalComponentAnalysis(images);

% Generate Model from Dataset
modelKNN = KNNTraining(trainPCA, labels);

% Plot Figure 
figure('NumberTitle', 'off', 'Name', 'KNN Main DR'), hold on
colours= ['r.'; 'g.'; 'b.'; 'k.'; 'y.'; 'c.'; 'm.'; 'r+'; 'g+'; 'b+'; 'k+'; 'y+'; 'c+'; 'm+'];
count=0;

for i=min(labels):max(labels)
    count = count+1;
    indexes = find (labels == i);
    plot3(trainPCA(indexes,1),trainPCA(indexes,2),trainPCA(indexes,3),colours(count,:));
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

kValue = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 31, 41, 51, 61, 71];

% Load Dataset
[test_images, test_labels] = loadFaceImages('face_test.cdataset', sampling);
indexesNegatives = find (test_labels == -1);
indexesPositives = find (test_labels == 1);

test_images = [test_images(indexesNegatives, :); test_images(indexesPositives,:)];
test_labels = [test_labels(indexesNegatives); test_labels(indexesPositives)];

% For each testing image, obtain a prediction based on the trained model

for k = 1:numel(kValue)
    
        for i=1:size(test_images,1)
            
            testIm= test_images(i,:);
            testIm = (testIm - meanX) * eigenVectors;
            classificationResult(i,1) = KNNTesting(testIm, modelKNN, kValue(k));
            
        end
    
    fprintf('KNN with k=%d\n', kValue(k));
    
    % Create the Confusion Matrix, Print all of the results
    createConfusionMatrix(test_labels, classificationResult);
    
end


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
