clear all
close all

Dim_Reduce = 0;

% Load all images & labels
[images,labels] = loadFaceImages('all_faces.cdataset',1);
gaborImages = getGabor(images)

% Fit SVM model using inbuilt Matlab function
if Dim_Reduce == 1
    [~,~,~,TrainPCA] = PrincipalComponentAnalysis(gaborImages);
    modelSVM = fitcsvm(TrainPCA,labels);
else 
    modelSVM = fitcsvm(gaborImages,labels);
end

% Create cross-validated model 
cvmodelSVM = crossval(modelSVM,'Kfold',10);

% Using the cross_validated model, predict Class labels 
results = kfoldPredict(cvmodelSVM);

% Print out results
createConfusionMatrix(labels,results);