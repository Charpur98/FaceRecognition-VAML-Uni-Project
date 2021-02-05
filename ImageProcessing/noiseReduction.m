function Iout = noiseReduction(I)

    Img = imread(I);
    
    mask = 1/(2^2) * ones(2,2);
    Iout = conv2(Img, mask);
    
end