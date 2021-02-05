clear all
close all

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tic

sampling = 1;

addpath .\shared_Funcs\;
addpath .\KNN_Implement\;
addpath .\images\;

% Load Dataset
[images, labels] = loadFaceImages('face_train.cdataset', sampling);

% Perform training
trainGabor = getGabor(images);

% Apply PCA, add ndim, or leave blank so the function will evaluate the 
% number needed to represent 95% total variance
[eigenVectors,eigenvalues,meanX,trainPCA] = PrincipalComponentAnalysis(trainGabor);

% Generate Model from Dataset
modelGaborKNN = KNNTraining(trainPCA, labels);

% Plot Figure 
figure('NumberTitle', 'off', 'Name', 'KNN Gabor DR'), hold on
colours= ['r.'; 'g.'; 'b.'; 'k.'; 'y.'; 'c.'; 'm.'; 'r+'; 'g+'; 'b+'; 'k+'; 'y+'; 'c+'; 'm+'];
count=0;

for i=min(labels):max(labels)
    count = count+1;
    indexes = find (labels == i);
    plot3(trainPCA(indexes,1),trainPCA(indexes,2),trainPCA(indexes,3),colours(count,:));
end

% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TESTING STAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

kValue = [1, 3, 5, 7, 9, 11, 21, 41];

% Load Dataset
[test_images, test_labels] = loadFaceImages('face_test.cdataset', sampling);
indexesNegatives = find (test_labels == -1);
indexesPositives = find (test_labels == 1);

test_images = [test_images(indexesNegatives, :); test_images(indexesPositives,:)];
test_labels = [test_labels(indexesNegatives); test_labels(indexesPositives)];

% Get Gabor features for every image in Test dataset
testGabor = getGabor(test_images);

% For each testing image, obtain a prediction based on th   e trained model
for k = 1:numel(kValue)
    
    for i=1:size(test_images,1)

        testIm= testGabor(i,:);
        testIm = (testIm - meanX) * eigenVectors;
        classificationResult(i,1) = KNNTesting(testIm, modelGaborKNN, kValue(k));
        
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
