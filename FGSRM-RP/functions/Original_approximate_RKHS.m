function [y,a]=Original_approximate_RKHS(Kernel,G,k,lb,ylb,M)
%% The original RKHS
if ~isfield(G,'e')
   G=gsp_compute_fourier_basis(G);   
end
Uk=G.U(:,1:k);

Kernel_lb=Kernel(:,lb);
% M=lambda*Kernel+gamma*Kernel*G.L*Kernel; %% 由于M矩阵只需要计算一次，故将其移到函数外面。
%A=Uk'*(Kernel_lb*Kernel_lb'+M)*Uk;
A=(Kernel_lb'*Uk)'*(Kernel_lb'*Uk)+Uk'*M*Uk;
b=Uk'*(Kernel_lb*ylb);
%c=A\b;%�ȼ۾������棨α�棩����Ľ�
c=pinv(A)*b;
a=Uk*c;
y=Kernel*a;

end