function y=Gaussian_KernelGramM(data_point,sigma)
%% Gaussian_Kernel gram matrix, 
%% sigma is the parameter, 
%% data denotes the set of feature vectors (each column is a feature vector)
n=size(data_point,2);
y=zeros(n);
for i=1:n
    for j=i+1:n
        y(i,j)=Gaussian_Kernelfunc(data_point(:,i),data_point(:,j),sigma);
    end
end
y=y+y'+eye(n);
end