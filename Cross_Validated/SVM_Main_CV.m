clear all
close all

Dim_Reduce = 0;

% Load all images & labels
[images,labels] = loadFaceImages('all_faces.cdataset',1);

% Fit SVM model using inbuilt Matlab function
if Dim_Reduce == 1
    [~,~,~,TrainPCA] = PrincipalComponentAnalysis(images);
    modelSVM = fitcsvm(TrainPCA,labels);
else 
    modelSVM = fitcsvm(images,labels);
end

% Create cross-validated model 
cvmodelSVM = crossval(modelSVM,'Kfold',10);

% Using the cross_validated model, predict Class labels 
results = kfoldPredict(cvmodelSVM);

% Print out results
createConfusionMatrix(labels,results);