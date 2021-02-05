function [nmsBoxes] = NonMaxSuppression(boundingBoxes, threshold)
    % Gets the individual points
    x1 = boundingBoxes(:,1);
    y1 = boundingBoxes(:,2);
    x2 = boundingBoxes(:,3);
    y2 = boundingBoxes(:,4);
    s = boundingBoxes(:,end);

    % Works out the area for each of the boundingBoxes
    area = (x2-x1+1) .* (y2-y1+1);
    
    % Gets the indexes of all of the bounding boxes
    [~, idxs] = sort(s);
    
    % Initlizes the list of bounding boxes to pick
    pickedBoxes = [];
    
    while ~isempty(idxs)
        % Gets the index of the last in the array
        last = length(idxs);
        
        % Gets the index value
        i = idxs(last);
 
        % Adds the index to the pick list
        pickedBoxes = [pickedBoxes; i];
        
        % Initlizes the list of boxes to delete
        delete = [last];
        
        % Loops through the array of remaining 
        for index = 1:last-1
            % Gets the index of the bounding box to check
            j = idxs(index);
            
            % Works out the largest coordinates for the start of
			% the bounding box and the smallest coordinates
			% for the end of the bounding box
            xx1 = max(x1(i), x1(j));
            yy1 = max(y1(i), y1(j));
            xx2 = min(x2(i), x2(j));
            yy2 = min(y2(i), y2(j));
            
            % Makes sure width and height of bounding boxes are greater
            % than 0
            w = xx2-xx1+1;
            h = yy2-yy1+1;
            if w > 0 && h > 0
                % Works out the overlap area
                overlapArea = w * h / area(j);
                
                % If the overlap area is greater than the threshold then
                % add to deete list.
                if overlapArea > threshold
                    delete = [delete; index];
                end
            end
        end
        
        % Deletes the indexes from idxs that are in the delete list
        idxs(delete) = [];
    end 
    
    % Debug Message
    fprintf("Number of faces found after NMS: %d\n", length(pickedBoxes));
    
    % Goes through and finds all the coordinates for the selected bounding
    % boxes and stores them in the array to be returned, nmsBoxes
    nmsBoxes = zeros(length(pickedBoxes), 4);
    for i = 1:length(pickedBoxes)
        nmsBoxes(i, 1) = boundingBoxes(pickedBoxes(i), 1);
        nmsBoxes(i, 2) = boundingBoxes(pickedBoxes(i), 2);
        nmsBoxes(i, 3) = boundingBoxes(pickedBoxes(i), 3);
        nmsBoxes(i, 4) = boundingBoxes(pickedBoxes(i), 4);
    end
end