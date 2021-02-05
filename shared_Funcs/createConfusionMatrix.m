function [confusionMatrix, values] = createConfusionMatrix(labels, predictions)
    numIms = length(predictions);
    facesPredicted = sum(predictions==1);
    % Create the Comparison table, comparing the actual labels to the
    % predictions made by the classifier
    comparisonTable = (labels==predictions);
    
    fprintf('\nPrediction Results:\n%d were identified as faces \n%d were identified as non-face\n', facesPredicted, numIms-facesPredicted);
    
    % Use Built In Confusion Matrix function to get TP, FP, TN and FN vals
    Matrix = confusionmat(labels, predictions);
    TP = Matrix(2,2);
    FP = Matrix(1,2);
    FN = Matrix(2,1);
    TN = Matrix(1,1);
    
    % Determine the overall correctness and incorrectness of the classification
    overall_correct = TP + TN;
    overall_incorrect = FP + FN;
    
    % True Positives + True Negatives
    accuracy = overall_correct/length(comparisonTable);
    fprintf('\n%d Images were correctly classified\n', overall_correct)
    % TP & TN
    fprintf('TP: %d Faces\n', TP)
    fprintf('TN: %d Non faces\n', TN)
    
    % Error Rate (FN + FP) / NInstances
    Error_rate = overall_incorrect/length(comparisonTable);
    fprintf('\n%d Images were incorrectly classified\n', overall_incorrect)
    % FP & FN
    fprintf('FP: %d Faces\n', FP)
    fprintf('FN: %d Non faces\n', FN)

    rates.recall = TP / (TP+FN);
    rates.precision = TP / (TP+FP);
    rates.specificity = TN / (TN+FP);
    rates.f1 = 2*TP / (2*TP + FN + FP);
    rates.falseAlarm = FP / (FP + TN);

    fprintf('\nAccuracy of model: %.2f%%\n', 100*accuracy);
    fprintf('Recall Rate of model: %.2f%%\n', 100*rates.recall);
    fprintf('Precision Rate of model: %.2f%%\n', 100*rates.precision);
    fprintf('Specifity Rate of model: %.2f%%\n', 100*rates.specificity);
    fprintf('F1 Rate of model: %.2f%%\n', 100*rates.f1);
    fprintf('False Alarm Rate of model: %.2f%%\n', 100*rates.falseAlarm);
    fprintf('-----------------------------------\n')
end