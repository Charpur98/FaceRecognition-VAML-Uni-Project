function [contrastedImg] = changeContrast(filename)
    
    Img = imread(filename);
    m = 1.7;
    c = -50;
    
    lut = (1:256);
    for j = 1:length(lut)
        if j < (-c/m)
                    lut(j) = 0;
                else if j > ((255 - c)/m)
                    lut(j) = 255;
                else
                    lut(j) = ((j-1)*m) + c;
         end
    end

    Lut = uint8(lut);
    contrastedImg = intlut(Img,Lut);

end
