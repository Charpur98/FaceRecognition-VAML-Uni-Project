clear all
close all

kValue = [3,5,7,9,11];
Dim_Reduce = 0;

% Load all images & labels
[images,labels] = loadFaceImages('all_faces.cdataset',1);
gaborImages = getGabor(images);

% If Dimensional Reduction Flag is set to 1, PCA will be carried out
if Dim_Reduce == 1
    [~,~,~,TrainPCA] = PrincipalComponentAnalysis(gaborImages);
end

for k=1:numel(kValue)
    % Fit KNN model using inbuilt Matlab function
    if Dim_Reduce == 1
        modelKNN = fitcknn(TrainPCA,labels,'NumNeighbors',kValue(k));
    else
        modelKNN = fitcknn(gaborImages,labels,'NumNeighbors',kValue(k));
    end
    
    % Create cross-validated model
    cvmodelKNN = crossval(modelKNN,'Kfold',10);

    % Using the cross_validated model, predict Class labels 
    results = kfoldPredict(cvmodelKNN);

    fprintf('KNN with K equal to %d \n', kValue(k));
    createConfusionMatrix(labels,results);
end