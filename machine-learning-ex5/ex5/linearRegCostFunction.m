function [J, Gradient] = linearRegCostFunction(X, y, Theta, Lambda)
%LINEARREGCOSTFUNCTION Compute cost and Gradientient for regularized linear 
%regression with multiple variables
%   [J, Gradient] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the Gradientient in Gradient

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
Gradient = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and Gradientient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and Gradient to the Gradientient.
%

Y = y;
H = X * Theta;

cost_main_term           = (1/(2*m))      * ( sum( (H - Y) .^ 2 ));
cost_regularization_term = (Lambda/(2*m)) * ( sum( (Theta(2:end)) .^ 2 ));

J = cost_main_term + cost_regularization_term;

Linear_partial_derivative = (1 / m) * sum((H - Y) .* X);
Partial_derivative_regularization_term = (Lambda / m) * [0 ; Theta(2:end)]';

Gradient = Linear_partial_derivative + Partial_derivative_regularization_term;


% =========================================================================

Gradient = Gradient(:);

end
