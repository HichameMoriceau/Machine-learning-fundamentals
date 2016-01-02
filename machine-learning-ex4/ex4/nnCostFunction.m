function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta_1 and Theta_2, the weight matrices
% for our 2 layer neural network
Theta_1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta_2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta_1_grad = zeros(size(Theta_1));
Theta_2_grad = zeros(size(Theta_2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta_1_grad and Theta_2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta_1 and Theta_2 in Theta_1_grad and
%         Theta_2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% why is this necessary ?!
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

%% FORWARD %%
% adding bias layer 1
A1 = [ones(m,1) X];

% forward to second layer
Z2 = A1 * Theta_1';
A2 = sigmoid(Z2);

% adding bias layer 2
A2 = [ones(m,1) A2];

% forward to third layer
Z3 = A2 * Theta_2';
A3 = sigmoid(Z3);
hypothesis = predict = A3;

[predict_max, index_max] = max(predict, [], 2);
p = index_max;

regularization_penalty = (lambda / (2*m)) * (sum(sum(Theta_1(:, 2:end) .^ 2)) + sum(sum(Theta_2(:, 2:end) .^ 2)));

J = (1/m) * sum(sum((-Y) .* log(hypothesis) - (1-Y) .* log(1 - hypothesis)));
J = J + regularization_penalty;

% Computing 'small deltas'
Sigma_3 = A3 - Y;
Sigma_2 = (Sigma_3 * Theta_2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);

Delta_1 = Sigma_2' * A1;
Delta_2 = Sigma_3' * A2;

%% Compute partial derivatives (ignoring bias terms)
## Theta_1_grad = (1/m)*(Delta_1 + lambda * [zeros(size(Theta_1, 1), 1), Theta_1(:, 2:end)]);
## Theta_2_grad = (1/m)*(Delta_2 + lambda * [zeros(size(Theta_2, 1), 1), Theta_2(:, 2:end)]);


Theta_1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta_1,1), 1) Theta_1(:, 2:end)];
Theta_2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta_2,1), 1) Theta_2(:, 2:end)];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta_1_grad
%               and Theta_2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta_1_grad(:) ; Theta_2_grad(:)];


end
