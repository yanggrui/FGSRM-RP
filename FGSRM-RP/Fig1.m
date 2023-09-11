%% Fig. 1

close all
clear all
addpath('functions','data','results');
% format short

N = 500;  % Number of vertices
Ns = 2*N; % Number of sample

% param.method = 'cheby';
% param.order = 30;
% param.Nrandom = 10;
% param.Nfilt = 50;

coords=rand(N,8);
param.show_edges = 1;
param.k=20;   %% k-nearest graph
G = gsp_nn_graph(coords,param);
% G = gsp_compute_fourier_basis(G);

G=gsp_estimate_lmax(G);
g= @(x) sin(4*pi*x/G.lmax).*(x<G.lmax/4);
w = randn(N,Ns);
s =gsp_filter(G,g,w);  %% generate the true sorce matrix based on the filter g



rate_set=0.01:0.01:0.05
sigma=4;
lambda=0.022;
gamma=0.05;

run=0; %% set run=1 to rerun the experiment

if run

    Kernel=Gaussian_KernelGramM(coords',sigma); %

    Running_time=zeros(6,length(rate_set));
    MAE=zeros(6,length(rate_set));

    for m=1:length(rate_set)

        rate_m=rate_set(m);
        M_scores=zeros(size(s));
        for j=1:size(M_scores,2)
            mark=rand(N,1)>(1-rate_m);
            M_scores(:,j)=s(:,j).*mark;
        end
        %% Obtain the Score Matrix in the test set
        M_scores_test=s.*(~mark);
        id_test= M_scores_test>0;
        %% Eliminate the columns without known labels
        for j=1:size(id_test,2)
            if ~any(M_scores(:,j))
                id_test(:,j)=zeros(size(id_test(:,j)));
                M_scores_test(:,j)=zeros(size(id_test(:,j)));
            end
        end
        fprintf('* The number of the prediction entries: %d.\n',sum(sum(id_test)));


        %% GBa10 method
        k_b=10;
        tic
        G=gsp_compute_fourier_basis(G);
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                %       G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(1,m)=toc
        MAE(1,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa20 method
        k_b=20;
        tic
        G=gsp_compute_fourier_basis(G);
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                %        G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(2,m)=toc
        MAE(2,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa50 method
        k_b=50;
        tic
        G=gsp_compute_fourier_basis(G);
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                %        G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(3,m)=toc
        MAE(3,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa100 method
        k_b=100;
        tic
        G=gsp_compute_fourier_basis(G);
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                %        G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(4,m)=toc
        MAE(4,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))


        %% Ori method
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                M_Ori=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                M_pre(:,j)=Original_RKHS(Kernel,G,lb,ylb,M_Ori);
            end
        end
        Running_time(5,m)=toc
        MAE(5,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))


        %% the proposed method
        M_pre=zeros(size(M_scores_test));
        flag=0;
        tic
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                %           T=inv(lambda*eye(G.N)+gamma*G.L*Kernel);
                %           R=Kernel*T;
                R=Kernel/(lambda*speye(G.N)+gamma*G.L*Kernel);
            end
            if ~isempty(lb)
                flag=flag+1;
                RLL=R(lb,lb);
                d=(RLL+eye(length(lb)))\ylb;% Solve the linear equations
                M_pre(:,j)=R(:,lb)*d;
            end
        end
        flag
        Running_time(6,m)=toc
        MAE(6,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

    end

    save('results\Fig1_MAE.mat','MAE')
    save('results\Fig1_Running_time','Running_time')

else

    load('results\Fig1_MAE.mat')
    load('results\Fig1_Running_time')

end


% MAE
% MAE=vpa(MAE,3)
figure(1)

%semilogy
semilogy(rate_set,MAE(1,:),'-*',rate_set,MAE(2,:),'-o',rate_set,MAE(3,:),'-s', ...
    rate_set,MAE(4,:),'-d',rate_set,MAE(5,:),'-x',rate_set,MAE(6,:),'r-p','linewidth',1.5)
xlabel('Ratio','Fontsize',16)
ylabel('\bf{MAE}','Fontsize',16,'Color','k')
legend('\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}', ...
    '\bf{Prop.}','Location','northeast');
ax=gca;
% axis([0.01 0.05 0.175 0.25]);
ax.FontName='Times New Roman';
ax.FontSize = 20;
tick={'1%','2%','3%','4%','5%'};
%set(l,'Fontsize',12)




figure(2)
%semilogy
plot(rate_set,Running_time(1,:),'-*',rate_set,Running_time(2,:),'-o', ...
    rate_set,Running_time(3,:),'-s',rate_set,Running_time(4,:),'-d', ...
    rate_set,Running_time(5,:),'-x',rate_set,Running_time(6,:),'r-p','linewidth',1.5)
xlabel('Ratio','Fontsize',16)
ylabel('Computation Time (s)','Fontsize',16,'Color','k')
legend('\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}', ...
    '\bf{Prop.}','Location','northwest');
%set(l,'Fontsize',12)
ax=gca;
% axis([0.01 0.05 0 15]);
ax.FontName='Times New Roman';
ax.FontSize = 20;
tick={'1%','2%','3%','4%','5%'};

print('-f1','results\Syn1','-djpeg')
print('-f2','results\Syn2','-djpeg')
print('-f1','results\Syn1','-dpng')
print('-f2','results\Syn2','-dpng')