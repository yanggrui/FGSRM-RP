%% Fig. 3

close all
clear all
addpath('functions','data','results');

run=0;   %% set run=1 to rerun the experiment
if run
    N=500;  % Number of vertices
    param.show_edges = 1;
    param.k=20;   %% k-nearest graph
    sigma=4;
    
    coords=rand(N,8);
    Kernel=Gaussian_KernelGramM(coords',sigma);
    G= gsp_nn_graph(coords,param);
    G=gsp_estimate_lmax(G);
    g= @(x) sin(4*pi*x/G.lmax).*(x<G.lmax/4);
    
    rate_m=0.05;
    lambda=0.022;
    gamma=0.05;
    
    Ns=1001;
    
    w = randn(N,Ns);
    s =gsp_filter(G,g,w);
    
    M_scores=zeros(size(s));
    k=2;
    mark=rand(N,k*Ns)<=rate_m;
    idx=find(sum(mark)<=5);
    if ~isempty(idx)
        mark(:,idx)=[];
    end
    
    while size(mark,2)<Ns
        k=k+1;
        mark=rand(N,k*Ns)<=rate_m;
        if ~isempty(idx)
            mark(:,idx)=[];
        end
    end
    mark=mark(:,1:Ns);
    
    M_scores(find(mark))=s(find(mark));
    %% Obtain the Score Matrix in the test set
    M_scores_test=s.*(~mark);
    id_test= M_scores_test>0;
    
    %% GBa10 method
    k_b=10;
    M_pre=zeros(size(M_scores_test));
    tic
    G=gsp_compute_fourier_basis(G);
    lb=find(M_scores(:,1));
    ylb=M_scores(lb,1);
    M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
    [M_pre(:,1),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
    FirstCPUTime=toc;
    GBa10time=[];
    for j=2:size(M_pre,2)
        tic
        lb=find(M_scores(:,j));
        ylb=M_scores(lb,j);
        [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
        GBa10time=[GBa10time;toc];
    end
    
    
    %% Ori method
    M_pre=zeros(size(M_scores_test));
    tic
    lb=find(M_scores(:,j));
    ylb=M_scores(lb,j);
    M_Ori=lambda*Kernel+gamma*Kernel*G.L*Kernel;
    M_pre(:,1)=Original_RKHS(Kernel,G,lb,ylb,M_Ori);
    FirstCPUTime=[FirstCPUTime,toc];
    
    Oritime=[];
    for j=2:size(M_pre,2)
        tic
        lb=find(M_scores(:,j));
        ylb=M_scores(lb,j);
        
        M_pre(:,j)=Original_RKHS(Kernel,G,lb,ylb,M_Ori);
        Oritime=[Oritime;toc];
    end
    
    %% the proposed method
    M_pre=zeros(size(M_scores_test));
    
    tic
    lb=find(M_scores(:,j));
    ylb=M_scores(lb,j);
    R=Kernel/(lambda*speye(G.N)+gamma*G.L*Kernel);
    RLL=R(lb,lb);
    d=(RLL+eye(length(lb)))\ylb;% Solve the linear equations
    M_pre(:,1)=R(:,lb)*d;
    FirstCPUTime=[FirstCPUTime,toc];
    
    Proptime=[];
    for j=2:size(M_pre,2)
        tic
        lb=find(M_scores(:,j));
        ylb=M_scores(lb,j);
        RLL=R(lb,lb);
        d=(RLL+eye(length(lb)))\ylb;% Solve the linear equations
        M_pre(:,j)=R(:,lb)*d;
        Proptime=[Proptime;toc];
    end
    
    Times.FirstCPUTime=FirstCPUTime;
    Times.GBa10time=GBa10time;
    Times.Oritime=Oritime;
    Times.Proptime=Proptime;
    save('results\Times.mat','Times');
else
    load('results\Times.mat');
    FirstCPUTime=Times.FirstCPUTime;
    GBa10time=Times.GBa10time;
    Oritime=Times.Oritime;
    Proptime=Times.Proptime;
end

CPUTime=[Times.GBa10time,Times.Oritime,Times.Proptime];
figure;
boxplot(CPUTime);
hold on;
plot(1:3,Times.FirstCPUTime/10,'bo-','linewidth',1.5);
xlabel('Models','Fontsize',16);
ylabel('Computation Time (s)','Fontsize',16,'Color','k');
ax=gca;
ax.FontName='Times New Roman';
ax.FontSize = 12;
xticks([1 2 3]);
xticklabels({'GBa10','Ori','Prop.'});


print('-f1','results\Syn5','-djpeg')
print('-f1','results\Syn5','-dpng')


disp('CPU time:---GBa10---------Ori.---------Prop.---');   
disp(['The first:  ',num2str(FirstCPUTime(1)),'     ',num2str(FirstCPUTime(2)),...
    '     ',num2str(FirstCPUTime(3))]);
disp(['  mean:   ',num2str(mean(GBa10time)),'    ',num2str(mean(Oritime)),...
    '   ',num2str(mean(Proptime))]);
disp(['median:   ',num2str(median(GBa10time)),'    ',num2str(median(Oritime)),...
    '   ',num2str(median(Proptime))]);
disp(['   max:   ',num2str(max(GBa10time)),'      ',num2str(max(Oritime)),...
    '     ',num2str(max(Proptime))]);
disp(['   min:   ',num2str(min(GBa10time)),'    ',num2str(min(Oritime)),...
    '   ',num2str(min(Proptime))]);
     



