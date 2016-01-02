function centroids = computeCentroids(X, C, K)
%COMPUTECENTROIDS returs the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% compute mean of each cluster
for(i=1:K)
  nb_example_in_cluster = 0;
  for(j=1:m)
    % if training set example j belongs to same cluster i
    if(C(j) == i)
      % consider it in for mean calculation
      centroids(i,:) = centroids(i,:) + X(j,:);
      nb_example_in_cluster = nb_example_in_cluster + 1;
    end;
  end;
  % compute average
  mean = centroids(i,:) / nb_example_in_cluster;
  centroids(i,:) = mean;
end;





% =============================================================


end

