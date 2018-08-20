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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% Add the row for bias.
X = [ones(m, 1) X];
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ------------Calculate the cost.--------------------------

% covert label-based y to vector-based y.
y1 = zeros(m, num_labels);
for row = 1:m;
	y1(row, y(row)) = 1;
end;

a1 = sigmoid(X * Theta1');
m_a1 = size(a1, 1);
hx = sigmoid([ones(m_a1, 1) a1] * Theta2');
J = (-1.*y1).*log(hx) - (1.-y1).*log(1.-hx);
J = sum(J(:)) / m;
J = J + ( ( sum((Theta1.^2)(:)) - sum((Theta1(:,1).^2)(:)) + sum((Theta2.^2)(:)) - sum((Theta2(:,1).^2)(:)) ) * lambda) / (2*m);


% ----------------Calculate the gradient.---------------------------------------------
for t = 1:m;
	z2 = X(t,:) * Theta1';
	a2 = sigmoid(z2);
	m_a2 = size(a2, 1);
	a3 = sigmoid([ones(m_a2, 1) a2] * Theta2');
	delta3 = a3.-y1(t,:);
	delta2 = Theta2'*delta3'.*[ones(m_a2, 1) sigmoidGradient(z2)]';
	Theta2_grad = Theta2_grad + delta3'*[ones(m_a2, 1) a2];
	Theta1_grad = Theta1_grad + delta2(2:end)*X(t,:);
end;

Theta1_grad = Theta1_grad./m;
Theta2_grad = Theta2_grad./m;

% ================Add regularization.=========================================================
Theta1_grad = [Theta1_grad(:,1) Theta1_grad(:,2:end).+(Theta1(:,2:end).*lambda./m)];
Theta2_grad = [Theta2_grad(:,1) Theta2_grad(:,2:end).+(Theta2(:,2:end).*lambda./m)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
