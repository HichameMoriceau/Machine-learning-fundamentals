function [J, gradient] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% calculating cost J 
hypothesis = sigmoid(X * theta);
errors = y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis);

cost = (-1 / m) * sum(errors);

% including the regularization
cost_with_regularization = cost + (lambda/(2*m)) * sum(theta(2:end) .^ 2);

J = cost_with_regularization;

gradient = (1 / m) * sum( (hypothesis - y) .* X ) + (lambda/m) * [0;theta(2:end)]';

% =============================================================

end
