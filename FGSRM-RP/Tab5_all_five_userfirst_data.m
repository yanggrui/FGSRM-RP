%% Table 5 all five user-first data

close all
clear all


load('data\Ratings_all_five_userfirst.mat')

run=0;  %% set run=1 to rerun the experiment

if run

    sigma=4;
    lambda=0.022;
    gamma=0.05;
    k=20;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%     user-based    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize the outputs
    Running_time=zeros(1,6);
    MAE=zeros(1,6);
    ratio=0.01; %%

    M=Ratings_five_userfirst.data;  %% The row is item and the column is user
    M=M';
    fprintf('* The number of items and users: %d and %d.\n', size(M,1),size(M,2))
    id_all_known=M>0;
    %     sum(id_all_known(:))/(size(M,1)*size(M,2))
    ntest=sum(id_all_known(:))-fix(size(M,1)*size(M,2)*ratio);
    [test_set,M_scores]=GetTestSet1(M,ntest);

    fprintf('* The number of the known entries and ratio: %d and %d.\n',...
        sum(sum(M_scores>0)),sum(sum(M_scores>0))/(size(M,1)*size(M,2)));

    %% Obtain the Score Matrix in the test set
    M_scores_test=zeros(size(M));
    for j=1:size(test_set,1)
        M_scores_test(test_set(j,1),test_set(j,2))=test_set(j,3);
    end
    size(M_scores_test)
    size(M_scores)
    id_test= M_scores_test>0;
    %% Eliminate the columns without known labels
    for j=1:size(id_test,2)
        if ~any(M_scores(:,j))
            id_test(:,j)=zeros(size(id_test(:,j)));
        end
    end
    sum(sum(id_test))
    fprintf('* The number of the prediction entries: %d.\n',sum(sum(id_test)));

    Feature_vectors=ConstrucionOfFeatureVectors(M_scores);

    Kernel=Gaussian_KernelGramM(Feature_vectors,sigma); % Gaussian gram kernel matrix

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
    Running_time(1)=toc
    MAE(1)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

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
    Running_time(2)=toc
    MAE(2)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

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
    Running_time(3)=toc
    MAE(3)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

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
    Running_time(4)=toc
    MAE(4)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))


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
    Running_time(5)=toc
    MAE(5)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))


    %% the proposed method
    tic
    M_pre=zeros(size(M_scores_test));
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
    Running_time(6)=toc
    MAE(6)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

    save('results\Tab5_userfirst_MAE_userbased.mat','MAE')
    save('results\Tab5_userfirst_Running_time_userbased.mat','Running_time')

else
    load('results\Tab5_userfirst_MAE_userbased.mat')
    load('results\Tab5_userfirst_Running_time_userbased.mat')
end

userfirst_userbased.Running_time=Running_time;


figure(1)
yyaxis left
bar(MAE)
ylim_max=max(MAE);ylim_min=min(MAE);
ylim([ylim_min-0.01 ylim_max+0.02])
ylabel('\bf{MAE}','Fontsize',16,'Color','k')
for i=1:length(MAE)
    text(i,MAE(i),num2str(MAE(i)),'VerticalAlignment','bottom',...
        'HorizontalAlignment','center','FontSize',12,'color','k')
end


yyaxis right
hLine=semilogy(1:length(Running_time),Running_time);
set(hLine,'color',[0,1,0],'LineWidth',2,'Marker','s','MarkerSize',10,...
    'MarkerFace',[0,1,0])
for i=1:length(Running_time)
    text(i,Running_time(i),num2str(Running_time(i)),'VerticalAlignment','bottom',...
        'HorizontalAlignment','left','FontSize',12,'color','r')
end
xlabel('Prediction Methods','Fontsize',16)
ylabel('Computation  Time (s)','Fontsize',16,'Color','k')
xlim([0.2,6.7])
tick={'\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}','\bf{Prop.}'};
set(gca,'XTickLabel',tick)
l=legend('\bf{MAE}','Time','Location','best');
set(l,'Fontsize',16);
saveas(gcf,'results\Tab5_userfirst_userbased.fig')
saveas(gcf,'results\Tab5_userfirst_userbased.jpg')
saveas(gcf,'results\Tab5_userfirst_userbased.png')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%     item-based    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% use the same parameter
%% Initialize the outputs
if run
    Running_time=zeros(1,6);
    MAE=zeros(1,6);
    %%% ratio=0.01; %%

    M=Ratings_five_userfirst.data;  %% The row is item and the column is user
    fprintf('* The number of items and users: %d and %d.\n', size(M,1),size(M,2))
    id_all_known=M>0;

    ntest=sum(id_all_known(:))-fix(size(M,1)*size(M,2)*ratio);
    [test_set,M_scores]=GetTestSet1(M,ntest);

    fprintf('* The number of the known entries and ratio: %d and %d.\n',...
        sum(sum(M_scores>0)),sum(sum(M_scores>0))/(size(M,1)*size(M,2)));

    %% Obtain the Score Matrix in the test set
    M_scores_test=zeros(size(M));
    for j=1:size(test_set,1)
        M_scores_test(test_set(j,1),test_set(j,2))=test_set(j,3);
    end
    size(M_scores_test)
    size(M_scores)
    id_test= M_scores_test>0;
    %% Eliminate the columns without known labels
    for j=1:size(id_test,2)
        if ~any(M_scores(:,j))
            id_test(:,j)=zeros(size(id_test(:,j)));
        end
    end
    fprintf('* The number of the prediction entries: %d.\n',sum(sum(id_test)));

    Feature_vectors=ConstrucionOfFeatureVectors(M_scores);

    Kernel=Gaussian_KernelGramM(Feature_vectors,sigma); % Gaussian gram kernel matrix

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
    Running_time(1)=toc
    MAE(1)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

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
    Running_time(2)=toc
    MAE(2)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

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
    Running_time(3)=toc
    MAE(3)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

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
    Running_time(4)=toc
    MAE(4)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))



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
    Running_time(5)=toc
    MAE(5)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))


    %% the proposed method
    tic
    M_pre=zeros(size(M_scores_test));
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
    Running_time(6)=toc
    MAE(6)=mean(abs(M_pre(id_test)-M_scores_test(id_test)))

    save('results\Tab5_userfirst_MAE_itembased.mat','MAE')
    save('results\Tab5_userfirst_Running_time_itembased.mat','Running_time')
else

    load('results\Tab5_userfirst_MAE_itembased.mat')
    load('results\Tab5_userfirst_Running_time_itembased.mat')

end
userfirst_itembased.Running_time=Running_time;

figure(2)
yyaxis left
bar(MAE)
ylim_max=max(MAE);ylim_min=min(MAE);
ylim([ylim_min-0.01 ylim_max+0.02])
ylabel('\bf{MAE}','Fontsize',16,'Color','k')
for i=1:length(MAE)
    text(i,MAE(i),num2str(MAE(i)),'VerticalAlignment','bottom',...
        'HorizontalAlignment','center','FontSize',12,'color','k')
end


yyaxis right
hLine=semilogy(1:length(Running_time),Running_time);
set(hLine,'color',[0,1,0],'LineWidth',2,'Marker','s','MarkerSize',10,...
    'MarkerFace',[0,1,0])
for i=1:length(Running_time)
    text(i,Running_time(i),num2str(Running_time(i)),'VerticalAlignment','bottom',...
        'HorizontalAlignment','left','FontSize',12,'color','r')
end
xlabel('Prediction Methods','Fontsize',16)
ylabel('Computation  Time (s)','Fontsize',16,'Color','k')
xlim([0.2,6.7])
tick={'\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}','\bf{Prop.}'};
set(gca,'XTickLabel',tick)
l=legend('\bf{MAE}','Time','Location','best');
set(l,'Fontsize',20)

saveas(gcf,'results\Tab5_userfirst_item_based.fig')
saveas(gcf,'results\Tab5_userfirst_item_based.jpg')
saveas(gcf,'results\Tab5_userfirst_item_based.png')









disp('*Speedup factor for user-based prediction in Netflix user first data:');
Running_time=userfirst_userbased.Running_time;
userfirst_userbased_SF=Running_time(1:5)/Running_time(6)

disp('*Speedup factor for item-based prediction in Netflix user first data:');
Running_time=userfirst_itembased.Running_time;
userfirst_itembased_SF=Running_time(1:5)/Running_time(6)

