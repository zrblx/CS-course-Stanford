function isFace = isFace( img )
% Decides if a given image is a human face
%   INPUT:
%       img - a grayscale image, size 120 x 100
%   OUTPUT:
%       isFace - should be true if the image is a human face, false if not.

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                     %
    %                       YOUR PART 4 CODE HERE                         %
    %                                                                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Design your own method to decide if the image is a human face
    %fisherface algorithm
    [rawFaceMatrix, imageOwner, imgHeight, imgWidth] = readInFaces();
    facenum = size(rawFaceMatrix,2);
    nonfacedir = dir('./nonfacedatabase/*.png');
    for i = 1:length(nonfacedir)
        img2 = double(rgb2gray(imread(['./nonfacedatabase/' nonfacedir(i).name])));
        rawFaceMatrix(:,i+facenum) = img2(:);
        imageOwner(i+facenum) = 0;
    end
    meanFace = sum(rawFaceMatrix,2)/size(rawFaceMatrix,2);
    A = rawFaceMatrix-meanFace;
    
    [~,col] = size(img);
    new_testImg = zeros(0);
    for y= 1:col
        new_testImg = [new_testImg;img(:,y)];
    end
    img = new_testImg-meanFace;
    
    numComponentsToKeep = 20;

    [prinComponents, weightCols] = fisherfaces(A,imageOwner,numComponentsToKeep);    
    prinComponents_t = prinComponents';
    %prinComponents_t size is numComponentsToKeep x sampleDimensionality 
   
    row = size(prinComponents_t,1);
    training_PCA = prinComponents_t(2:row,:)*A;

    test_weight = prinComponents_t(2:row,:)*img;

    
    [dist, indexofclosest] = indexOfClosestColumn(training_PCA, test_weight);
    isFace = boolean(imageOwner(indexofclosest));
   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [minDist, indexOfClosest] = indexOfClosestColumn(A, testColumn)
% Compares testColumn to each column in A and returns the distance and
% index of the closest column in A.
% INPUTS:
%   A: A matrix where each column is a data sample. Its size is
%   sampleDimensionality x numSamples.
%   testColumn: A column vector which needs to be compared to each column
%   in the matrix A.
% RETURNS:
%   minDist: The Euclidean distance between testColumn and its closest
%   column in A
%   closest: The index of the column in the matrix which has the lowest
%   Euclidean distance to testColumn.

    col = size(A,2);
    distance = zeros(1,col);
    for j=1:col
        diff = testColumn-A(:,j);
        distance(1,j) = sqrt(sumsqr(diff));
    end
    minDist = min(distance);
    [~,indexOfClosest] = find(distance == minDist);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                     %
    %                            END YOUR CODE                            %
    %                                                                     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

