function [contrastedImg] = changeContrast_HE(filename)
  
    Img = imread(filename);

    Lut = 1:256; 
   
    hist = imhist(Img);
    
    for j=1:length(hist)
        value = cumsum(hist(1:j));
        
        Lut(1,j) = max(0, round((value(end) / 255) -1));
    end
   
   % output value = max {0,(short)[256 * CH(input value) / (number of pixels in image)] – 1}
   
   Lut = uint8(Lut);
   contrastedImg = intlut(Img, Lut);

end
