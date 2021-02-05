function Iout = enhanceBrightness(Iin, c)

    Img = imread(Iin);

    lut = zeros(1, 256);

    for i = 1:length(lut)
        lut(i) = i - 1;
    end

    for j = 1:length(lut)
    if lut(j) < - c
       lut(j) = 0;
    else if lut(j) > 255 - c
            lut(j) = 255;
        else
            lut(j) = lut(j) + c;
        end
    end
    
    Lut = uint8(lut);
    Iout = intlut(Img, Lut);

    end
end
