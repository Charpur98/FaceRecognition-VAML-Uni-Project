function prediction = NNTesting(testImage, modelNN)

    distance = zeros(size(modelNN.neighbours,1),1);
    minDistance = 10000;

    for i=1:size(modelNN.neighbours,1) 
        distance(i) = EuclideanDistance(testImage, modelNN.neighbours(i,:));
        
        if (distance(i) < minDistance)
            minDistance = distance(i);
            prediction = modelNN.labels(i);
        end 
        
    end
    
end
