%% Fig. 2

close all
clear all
addpath('functions','data','results');
% format short
run=0;  %% set run=1 to rerun the experiment
if run
    N= [200,1000,2000,5000];  % Number of vertices
    Ns = 1000; % Number of sample
    
    param.show_edges = 1;
    param.k=20;   %% k-nearest graph
    sigma=4;
    
    rate_m=0.05
    lambda=0.022;
    gamma=0.05;
    
    
    Running_time=zeros(3,length(N));
    MAE=zeros(3,length(N));
    
    for m=1:length(N)
        coords=rand(N(m),8);
        Kernel=Gaussian_KernelGramM(coords',sigma);
        G= gsp_nn_graph(coords,param);
        G=gsp_estimate_lmax(G);
        g= @(x) sin(4*pi*x/G.lmax).*(x<G.lmax/4);
        w = randn(N(m),Ns);
        s =gsp_filter(G,g,w);
        
        k=2;
        mark=rand(N(m),k*Ns)<=rate_m;
        idx=find(sum(mark)<=5);
        if ~isempty(idx)
            mark(:,idx)=[];
        end
        
        while size(mark,2)<Ns
            k=k+1;
            mark=rand(N(m),k*Ns)<=rate_m;
            if ~isempty(idx)
                mark(:,idx)=[];
            end
        end
        mark=mark(:,1:Ns);
        
        M_scores=zeros(size(s));
        M_scores(find(mark))=s(find(mark));
        
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
            [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
        end
        Running_time(1,m)=toc
        MAE(1,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))
        
        %% Ori method
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                M_Ori=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            M_pre(:,j)=Original_RKHS(Kernel,G,lb,ylb,M_Ori);
        end
        Running_time(2,m)=toc
        MAE(2,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))
        
        
        %% the proposed method
        M_pre=zeros(size(M_scores_test));
        tic
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                %           T=inv(lambda*eye(G.N)+gamma*G.L*Kernel);
                %           R=Kernel*T;
                R=Kernel/(lambda*speye(G.N)+gamma*G.L*Kernel);
            end
            RLL=R(lb,lb);
            d=(RLL+eye(length(lb)))\ylb;% Solve the linear equations
            M_pre(:,j)=R(:,lb)*d;
        end
        Running_time(3,m)=toc
        MAE(3,m)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))
        
    end
    
 
    Exp2.MAE=MAE;
    Exp2.Running_time=Running_time;
    save('results\Exp2.mat','Exp2');
else
    load('results\Exp2.mat');
    MAE=Exp2.MAE;
    Running_time=Exp2.Running_time;
end

% MAE
% MAE=vpa(MAE,3)
figure(1)

%semilogy
plot(1:4,MAE(1,:),'-*',1:4,MAE(2,:),'-o',1:4,MAE(3,:),'-+','linewidth',1.5)
xlabel('Number of vertex','Fontsize',16)
ylabel('\bf{MAE}','Fontsize',16,'Color','k')
legend('\bf{GBa10}','\bf{Ori}', '\bf{Prop.}','Location','northeast');
ax=gca;
%axis([0.01 0.05 0.175 0.22]);
ax.FontName='Times New Roman';
ax.FontSize = 20;
xticks([1 2 3 4]);
xticklabels({'200','500','2000','5000'});
%set(l,'Fontsize',12)




figure(2)
semilogy(1:4,Running_time(1,:),'-*',1:4,Running_time(2,:),'-o', ...
    1:4,Running_time(3,:),'-+','linewidth',1.5)
xlabel('Number of vertex','Fontsize',16);
ylabel('Computation Time (s)','Fontsize',16,'Color','k')
legend('\bf{GBa10}','\bf{Ori}','\bf{Prop.}','Location','northwest');
%set(l,'Fontsize',12)
ax=gca;
%axis([0.01 0.05 0 15]);
ax.FontName='Times New Roman';
ax.FontSize = 20;
xticks([1 2 3 4]);
xticklabels({'200','500','2000','5000'});

print('-f1','results\Syn3','-djpeg')
print('-f2','results\Syn4','-djpeg')
print('-f1','results\Syn3','-dpng')
print('-f2','results\Syn4','-dpng')