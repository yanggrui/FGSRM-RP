%%  Fig. 4

close all
clear all

load('data\ml-100k\mat\u1base.mat')
load('data\ml-100k\mat\u1test.mat')
u{1}.base=u1base;
u{1}.test=u1test;
load('data\ml-100k\mat\u2base.mat')
load('data\ml-100k\mat\u2test.mat')
u{2}.base=u2base;
u{2}.test=u2test;
load('data\ml-100k\mat\u3base.mat')
load('data\ml-100k\mat\u3test.mat')
u{3}.base=u3base;
u{3}.test=u3test;
load('data\ml-100k\mat\u4base.mat')
load('data\ml-100k\mat\u4test.mat')
u{4}.base=u4base;
u{4}.test=u4test;
load('data\ml-100k\mat\u5base.mat')
load('data\ml-100k\mat\u5test.mat')
u{5}.base=u5base;
u{5}.test=u5test;


clear u1base u1test u2base u2test u3base u3test u4base u4test u51base u5test


run=0; %% set run=1 to rerun the experiment

if run
    n_U=943;m_I=1682;
    sigma=4;
    lambda=0.022;
    gamma=0.05;
    k=20;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%     user-based    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize the outputs
    Running_time=zeros(6,5);
    MAE=zeros(6,5);
    for r=1:5
        M_scores=zeros(n_U,m_I);
        %% Obtain the Score Matrix in the training set
        for j=1:size(u{r}.base,1)
            M_scores(u{r}.base(j,1),u{r}.base(j,2))=u{r}.base(j,3);
        end

        Feature_vectors=ConstrucionOfFeatureVectors(M_scores);
        fprintf('* The number of the known entries,radius: %d and %d.\n',sum(sum(M_scores>0)),sum(sum(M_scores>0))/(n_U*m_I));

        Kernel=Gaussian_KernelGramM(Feature_vectors,sigma); % Gaussian gram kernel matrix
        %% Obtain the Score Matrix in the test set
        M_scores_test=zeros(n_U,m_I);
        for j=1:size(u{r}.test,1)
            M_scores_test(u{r}.test(j,1),u{r}.test(j,2))=u{r}.test(j,3);
        end
        id_test= M_scores_test>0;

        %% Eliminate the columns without known labels
        for j=1:size(id_test,2)
            if ~any(M_scores(:,j))
                id_test(:,j)=zeros(size(id_test(:,j)));
                M_scores_test(:,j)=zeros(size(id_test(:,j)));
            end

        end
        fprintf('* The number of the prediction entries: %d.\n',sum(sum(id_test)));

        %% using sparse graph
        param.k=k;
        G=gsp_nn_graph(Feature_vectors',param);

        %% GBa10 method
        k_b=10;
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(1,r)=toc
        MAE(1,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa20 method
        k_b=20;
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(2,r)=toc
        MAE(2,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa50 method
        k_b=50;
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(3,r)=toc
        MAE(3,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa100 method
        k_b=100;
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(4,r)=toc
        MAE(4,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

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
        Running_time(5,r)=toc
        MAE(5,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))


        %% the proposed method
        M_pre=zeros(size(M_scores_test));
        tic
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                %           T=inv(lambda*eye(G.N)+gamma*G.L*Kernel);
                %           R=Kernel*T;
                R=Kernel/(lambda*eye(G.N)+gamma*G.L*Kernel);
            end
            if ~isempty(lb)
                RLL=R(lb,lb);
                d=(RLL+eye(length(lb)))\ylb;% Solve the linear equations
                M_pre(:,j)=R(:,lb)*d;
            end
        end
        Running_time(6,r)=toc
        MAE(6,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))
    end

    save('results\Com_u1_u5_MAE_user_based.mat','MAE')
    save('results\Com_u1_u5_Running_time_user_based.mat','Running_time')


else
    load('results\Com_u1_u5_MAE_user_based.mat')
    load('results\Com_u1_u5_Running_time_user_based.mat')
end
userbased.MAE=MAE;
running_time=mean(Running_time,2);
mae=mean(MAE,2);


figure(1)
yyaxis left
bar(mae)
ylim([0.7518 0.754])
% % axis auto
ylabel('\bf{MAE}','Fontsize',16,'Color','k')
for i=1:length(mae)
    text(i,mae(i),num2str(mae(i)),'VerticalAlignment','bottom',...
        'HorizontalAlignment','center','FontSize',12,'color','k')
end


yyaxis right
hLine=semilogy(1:length(running_time),running_time);
set(hLine,'color',[0,1,0],'LineWidth',2,'Marker','s','MarkerSize',10,...
    'MarkerFace',[0,1,0])
for i=1:length(running_time)
    text(i,running_time(i),num2str(running_time(i)),'VerticalAlignment','bottom',...
        'HorizontalAlignment','left','FontSize',12,'color','r')
end
xlabel('Prediction Methods','Fontsize',16)
ylabel('Computation  Time (s)','Fontsize',16,'Color','k')
xlim([0.2,6.7])
tick={'\bf{GBa10}','\bf{Gba20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}','\bf{Prop.}'};
set(gca,'XTickLabel',tick)
l=legend('\bf{MAE}','Time','Location','northwest');
set(l,'Fontsize',16);
% title('User-based','Fontsize',16)
saveas(gcf,'results\Com_u1_u5_user_based.fig')
saveas(gcf,'results\Com_u1_u5_user_based.jpg')
saveas(gcf,'results\Com_u1_u5_user_based.png')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%     item-based    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% use the same parameters
% % sigma=4;
% % lambda=0.022;
% % gamma=0.05;
% % k=20;
%% Initialize the outputs

if run
    Running_time=zeros(6,5);
    MAE=zeros(6,5);
    for r=1:5
        M_scores=zeros(m_I,n_U);
        %% Obtain the Score Matrix in the training set
        for j=1:size(u{r}.base,1)
            M_scores(u{r}.base(j,2),u{r}.base(j,1))=u{r}.base(j,3);
        end

        Feature_vectors=ConstrucionOfFeatureVectors(M_scores);
        fprintf('* The number of the known entries,radius: %d and %d.\n',sum(sum(M_scores>0)),sum(sum(M_scores>0))/(n_U*m_I));

        Kernel=Gaussian_KernelGramM(Feature_vectors,sigma); % Gaussian gram kernel matrix
        %% Obtain the Score Matrix in the test set
        M_scores_test=zeros(m_I,n_U);
        for j=1:size(u{r}.test,1)
            M_scores_test(u{r}.test(j,2),u{r}.test(j,1))=u{r}.test(j,3);
        end
        id_test= M_scores_test>0;
        %% Eliminate the columns without known labels
        for j=1:size(id_test,2)
            if ~any(M_scores(:,j))
                id_test(:,j)=zeros(size(id_test(:,j)));
                M_scores_test(:,j)=zeros(size(id_test(:,j)));
            end

        end

        fprintf('* The number of the prediction entries: %d.\n',sum(sum(id_test)));

        %% using sparse graph
        param.k=k;
        G=gsp_nn_graph(Feature_vectors',param);


        %% GBa10 method
        k_b=10;
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(1,r)=toc
        MAE(1,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa20 method
        k_b=20;
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(2,r)=toc
        MAE(2,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa50 method
        k_b=50;
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(3,r)=toc
        MAE(3,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

        %% GBa100 method
        k_b=100;
        tic
        M_pre=zeros(size(M_scores_test));
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                G=gsp_compute_fourier_basis(G);
                M_GBa=lambda*Kernel+gamma*Kernel*G.L*Kernel;
            end
            if ~isempty(lb)
                [M_pre(:,j),~]=Original_approximate_RKHS(Kernel,G,k_b,lb,ylb,M_GBa);
            end
        end
        Running_time(4,r)=toc
        MAE(4,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

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
        Running_time(5,r)=toc
        MAE(5,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))


        %% the proposed method
        M_pre=zeros(size(M_scores_test));
        tic
        for j=1:size(M_pre,2)
            lb=find(M_scores(:,j));
            ylb=M_scores(lb,j);
            if j==1
                R=Kernel/(lambda*eye(G.N)+gamma*G.L*Kernel);
            end
            if ~isempty(lb)
                RLL=R(lb,lb);
                d=(RLL+eye(length(lb)))\ylb;% Solve the linear equations
                M_pre(:,j)=R(:,lb)*d;
            end
        end
        Running_time(6,r)=toc
        MAE(6,r)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))
    end

    save('results\Com_u1_u5_MAE_item_based.mat','MAE')
    save('results\Com_u1_u5_Running_time_item_based.mat','Running_time')


else
    load('results\Com_u1_u5_MAE_item_based.mat')
    load('results\Com_u1_u5_Running_time_item_based.mat')
end
itembased.MAE=MAE;
running_time=mean(Running_time,2);
mae=mean(MAE,2);



figure(2)
yyaxis left
bar(mae)
ylim([0.73 0.736])
ylabel('\bf{MAE}','Fontsize',16,'Color','k')
for i=1:length(mae)
    text(i,mae(i),num2str(mae(i)),'VerticalAlignment','bottom',...
        'HorizontalAlignment','center','FontSize',12,'color','k')
end


yyaxis right
hLine=semilogy(1:length(running_time),running_time);
set(hLine,'color',[0,1,0],'LineWidth',2,'Marker','s','MarkerSize',10,...
    'MarkerFace',[0,1,0])
for i=1:length(running_time)
    text(i,running_time(i),num2str(running_time(i)),'VerticalAlignment','bottom',...
        'HorizontalAlignment','left','FontSize',12,'color','r')
end
xlabel('Prediction Methods','Fontsize',16)
ylabel('Computation  Time (s)','Fontsize',16,'Color','k')
xlim([0.2,6.7])
tick={'\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}','\bf{Prop.}'};
set(gca,'XTickLabel',tick)
l=legend('\bf{MAE}','Time','Location','northwest');
set(l,'Fontsize',16)
% title('Item-based','Fontsize',16)
saveas(gcf,'results\Com_u1_u5_item_based.fig')
saveas(gcf,'results\Com_u1_u5_item_based.jpg')
saveas(gcf,'results\Com_u1_u5_item_based.png')

userbased.MAE
disp('*std of prediction results of the prposed method for u1~u5 data:')
disp(['User-based:',num2str(std(userbased.MAE(6,:)))])
disp(['Item-based:',num2str(std(itembased.MAE(6,:)))])

