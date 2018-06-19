function featuresNorm = NormalizeFeatures(features)
% Normalize image features to have zero mean and unit variance. This
% normalization can cause k-means clustering to perform better.
%
% INPUTS
% features - An array of features for an image. features(i, j, :) is the
%            feature vector for the pixel img(i, j, :) of the original
%            image.
%
% OUTPUTS
% featuresNorm - An array of the same shape as features where each feature
%                has been normalized to have zero mean and unit variance.

    features = double(features);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE:                               %
%                                                                              %
%                HINT: The functions mean and std may be useful                %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [row,col,dim] = size(features);
   feature_mean = zeros(1,dim);
   feature_var = zeros(1,dim);
   
   for k=1:dim
      for i=1:row
         for j=1:col
             feature_mean(k) = feature_mean(k)+features(i,j,k);
         end
      end
      feature_mean(k) = feature_mean(k)/(row*col);
   end
   
   for k=1:dim
       for i=1:row
           for j=1:col
               feature_var(k) = feature_var(k)+(features(i,j,k)-feature_mean(k))^2;
           end
       end
       feature_var(k) = sqrt(feature_var(k)/(row*col-1));
   end
   
   featuresNorm = zeros(row,col,dim);
  
   for k=1:dim
       for i=1:row
           for j=1:col
               featuresNorm(i,j,k) = (features(i,j,k)-feature_mean(k))/feature_var(k);
           end
       end
       
   end        
               

end