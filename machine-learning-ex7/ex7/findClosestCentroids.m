function C = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
C = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

fprintf('Initiating findClosestCentroids.');

nb_examples = length(X);

for(i=1:nb_examples)
  min_distance = inf;
   % find closest centroid
   for(j=1:K)
     % compute distance (squared error) 
     % between this example and the current centroid
      distance_example_centroid = norm(X(i,:) - centroids(j,:)) ^ 2; 
      if(distance_example_centroid <= min_distance)
        closest_centroid = j;
        min_distance = distance_example_centroid;
      end;
   end;
   % assign cluster to current example
   C(i) = closest_centroid;
   %fprintf('Closest centroid for %f is %f', X(i), C(i));
end;

fprintf('Cluster assignment step finished.');



% =============================================================

end

