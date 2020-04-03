function [J, grad] = costFunctionReg(theta, X, y, lambda)
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

for i= 2:length(grad)
	 grad(i,1)=(sum((sigmoid(X*theta)-y).*X(:,i)))/m+lambda*theta(i,1)/m;

end

a =  -y'*log(sigmoid(X*theta));
b = -(1-y)'*log(1-sigmoid(X*theta));
J = (a+b)/m+lambda*sum(theta(2:length(grad),1).*theta(2:length(grad),1))/(2*m);
grad(1,1) = (sum(sigmoid(X*theta)-y))/m;

    





% =============================================================

end
