function [features] = getHOG(images)
    % Returns HOG features for each image in matrix
    numberOfImages = size(images,1);
    features=[];
    
    %Pixel values must be converted back to image format to use
    %hog_feature_vector function
    for i = 1:numberOfImages
        thisImg = reshape(images(i,:),27,18);
        thisImg = uint8(thisImg);
        gab = hog_feature_vector(thisImg);
        features=[features; gab];
    end
end