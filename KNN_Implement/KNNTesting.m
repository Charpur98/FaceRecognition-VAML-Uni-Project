function prediction = KNNTesting(testIm, modelNN, k)

    for i=1:size(modelNN.neighbours,1) 
        distance(i) = EuclideanDistance(testIm, modelNN.neighbours(i,:));     
    end
    
    distanceSorted = sort(distance);
    for i=1:k
        knn = distanceSorted(i);
        for j=1:size(modelNN.neighbours,1) 
            if (knn == distance(j))
                labels(i) = modelNN.labels(j);
            end
        end
    end
    
    prediction = mode(labels(:));
end