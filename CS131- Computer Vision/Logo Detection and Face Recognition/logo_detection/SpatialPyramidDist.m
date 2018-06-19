function D = SpatialPyramidDist( I1, I2, nbins )
%SpatialPyramidDist
%   Compute the spatial pyramid measure for the two given image
%   patches.
%
%Input:
%   I1: image patch 1
%   I2: image patch 2
%   nbins: number of bins for color histograms. Note this is unrelated to 
%   spatial partitioning of the image.
%
%Output:
%   D: spatial pyramid measure (a real number)
%
if nargin == 2
    nbins = 20;
end

numLevel = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE                                %
%                You should fill in D with the weighted distance               %
%                   between histograms of two entire images.                   %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = HistIntersectDist( I1, I2, nbins)/(2^numLevel);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                               END OF YOUR CODE                               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for l = 1 : numLevel
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE                                %
%             You should fill in numCells with the number of cells             %
%                           along x and y directions.                          %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    numCells = 2^l;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                               END OF YOUR CODE                               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for i = 1 : numCells
        for j = 1 : numCells
            
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE                                %
%               You should fill in x_lo1, x_hi1, y_lo1, y_hi1 to               %
%                    extract one cell of I1 in the pyramid.                    %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [row1,col1,~] = size(I1);
            x_len1 = floor(row1/numCells);
            y_len1 = floor(col1/numCells);
            
            x_lo1 = x_len1*(i-1)+1;
            x_hi1 = x_len1*i;
            y_lo1 = y_len1*(j-1)+1;
            y_hi1 = y_len1*j;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                               END OF YOUR CODE                               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE                                %
%               You should fill in x_lo2, x_hi2, y_lo2, y_hi2 to               %
%                    extract one cell of I2 in the pyramid.                    %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [row2,col2,~] = size(I2);
            x_len2 = floor(row2/numCells);
            y_len2 = floor(col2/numCells);
            
            x_lo2 = x_len2*(i-1)+1;
            x_hi2 = x_len2*i;
            y_lo2 = y_len2*(j-1)+1;
            y_hi2 = y_len2*j;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                               END OF YOUR CODE                               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
            
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE                                %
%You should increment D by the weighted distances between patches in this cell.%
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            D = D + HistIntersectDist( I1(x_lo1:x_hi1,y_lo1:y_hi1,:), I2(x_lo2:x_hi2,y_lo2:y_hi2,:), nbins)/(2^(numLevel+1-l));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                               END OF YOUR CODE                               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end
end

end

