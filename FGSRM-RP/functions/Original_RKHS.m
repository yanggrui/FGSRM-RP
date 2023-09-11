function y=Original_RKHS(Kernel,G,lb,ylb,M)
%% The original RKHS 
Kernel_lb=Kernel(:,lb);
% M=lambda*Kernel+gamma*Kernel*G.L*Kernel; %% M只需要计算一次，故移出函数外面
b=Kernel_lb*ylb;
% a=M\b;%�ȼ۾������棨α�棩����Ľ�
%a=pinv(M)*b;
a=(Kernel_lb*Kernel_lb'+M+1e-13*eye(G.N))\b;
y=Kernel*a;



end