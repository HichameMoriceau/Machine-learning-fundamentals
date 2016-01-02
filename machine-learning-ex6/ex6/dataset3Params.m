function [C, Sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
Sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% values of C and Sigma to try to use
values = [0.01 0.03 0.1 0.3 1 3 10 30];
error_min = inf;

for C_tmp = values
  for Sigma_tmp = values
      % define model
      model =svmTrain(X, y, C_tmp, @(x1, x2) gaussianKernel(x1, x2, Sigma_tmp));
      % Execute prediction
      predictions = svmPredict(model, Xval);
      % Compute prediction accuracy
      error       = mean(double(predictions ~= yval));
      if(error <= error_min)
         best_C     = C_tmp;
         best_Sigma = Sigma_tmp;
         error_min = error;
         fprintf('better combination discovered (C,sigma)=(%f,%f)\t(error=%f)', best_C, best_Sigma, error_min)
      end
  end
end

C     = best_C;
Sigma = best_Sigma;

fprintf('Best combination found: (C,Sigma)=(%f,%f)\t(error=%f)', C, Sigma,error_min );

% =========================================================================

end
