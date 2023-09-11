function y=Gaussian_Kernelfunc(x,y,sigma)
%% Gaussian kernel function
%% x and y denote the feature vector, and sigma is the parameter
y=exp(-norm(x-y)^2/(2*sigma^2));
end