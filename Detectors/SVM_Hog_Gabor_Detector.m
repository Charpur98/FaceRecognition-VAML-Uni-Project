function [boundingBoxes] = SVM_Hog_Gabor_Detector(hogModel, gaborModel, image, windowSize)
    row = 1;
    col = 1;
    [maxCol, maxRow] = size(image);

    windowMax = 1;
    for y = col:maxCol-windowSize(2)   
        for x = row:maxRow-windowSize(1)
            windowMax = windowMax+1;
        end
    end
    
    tic % Start Timer for Hog Detection
    hogResults = zeros(windowMax,1);
    hogBoundingBox = zeros(windowMax,4);
    windowNumber = 1;
    fprintf("Hog Detection Running\n");
    for y = col:maxCol-windowSize(2)   
        for x = row:maxRow-windowSize(1)
            po = [x, y, windowSize(1), windowSize(2)];
            img = imcrop(image, po); 
            gab = hog_feature_vector(img);
            hogResults(windowNumber) = SVMTesting(gab,hogModel);
            hogBoundingBox(windowNumber, 1) = x;
            hogBoundingBox(windowNumber, 2) = y;
            hogBoundingBox(windowNumber, 3) = x + windowSize(1);
            hogBoundingBox(windowNumber, 4) = y + windowSize(2);
            windowNumber = windowNumber+1;
        end
    end 
    hogBoundingBoxes = hogBoundingBox(hogResults == 1, :);
    fprintf("HOG found %d faces\n", length(hogBoundingBoxes));
    toc % End Timer for Hog Detection
    
    % -------------------------------------------------------------------
    
    tic % Start Timer for Gabor Detection
    gabResults = zeros(length(hogBoundingBoxes), 1);
    gabBoundingBox = zeros(length(hogBoundingBoxes), 4);
    fprintf("\nGabor Detection Running\n");
    for i = 1:length(hogBoundingBoxes)
        po = [hogBoundingBoxes(i, 1), hogBoundingBoxes(i, 2), windowSize(1), windowSize(2)];
        img = imcrop(image, po); 
        gab = gabor_feature_vector(img);
        gabResults(i) = SVMTesting(gab, gaborModel);
        gabBoundingBox(i, 1) = hogBoundingBoxes(i, 1);
        gabBoundingBox(i, 2) = hogBoundingBoxes(i, 2);
        gabBoundingBox(i, 3) = hogBoundingBoxes(i, 3);
        gabBoundingBox(i, 4) = hogBoundingBoxes(i, 4);
    end
 	boundingBoxes = gabBoundingBox(gabResults == 1, :);
    fprintf("Gabor found %d faces\n", length(hogBoundingBoxes));
    toc % End Timer for Gabor Detection
end