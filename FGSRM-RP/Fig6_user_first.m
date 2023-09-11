%% Fig.6 user-first



clear all
close all
load('data\ratings.mat')


run=0; %% set run=1 to rerun the experiment

ratio=0.01:0.005:0.05;
if run
    %%%%%%%%%%%%%%%%%%%%%       user-based    %%%%%%%%%%%%%%%%%%%%%%%%%
    sigma=4;
    lambda=0.022;
    gamma=0.05;
    k=20;

    %% Initialize the outputs
    Running_time=zeros(6,length(ratio));
    MAE=zeros(6,length(ratio)); %% five methods

    for r=1:length(ratio)
        ratio_r=ratio(r);
        abs_err_GBa10=[];abs_err_GBa20=[];abs_err_GBa50=[];
        abs_err_GBa100=[];abs_err_Ori=[];abs_err_Prop=[];
        for i=6:10
            %         if i<=5
            %            n_U=1000;
            %            m_I=1777;
            %         else
            %            n_U=1500;
            %            m_I=888;
            %          end
            M=ratings{i}.data;  %% The row is item and the column is user
            M=M';
            fprintf('* The number of users and items: %d and %d.\n', size(M,1),size(M,2))
            id_all_known=M>0;

            ntest=sum(id_all_known(:))-fix(size(M,1)*size(M,2)*ratio_r);
            [test_set,M_scores]=GetTestSet1(M,ntest);

            fprintf('* The number of the known entries and ratio: %d and %d.\n',...
                sum(sum(M_scores>0)),sum(sum(M_scores>0))/(size(M,1)*size(M,2)));

            %% Obtain the Score Matrix in the test set
            M_scores_test=zeros(size(M));
            for j=1:size(test_set,1)
                M_scores_test(test_set(j,1),test_set(j,2))=test_set(j,3);
            end
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
            t1=toc;
            Running_time(1,r)=Running_time(1,r)+t1;
            abs_err_GBa10=[abs_err_GBa10 abs(M_pre(id_test)-M_scores_test(id_test))'];


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
            t2=toc;
            Running_time(2,r)=Running_time(2,r)+t2;
            abs_err_GBa20=[abs_err_GBa20 abs(M_pre(id_test)-M_scores_test(id_test))'];


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
            t3=toc;
            Running_time(3,r)=Running_time(3,r)+t3;
            abs_err_GBa50=[abs_err_GBa50 abs(M_pre(id_test)-M_scores_test(id_test))'];

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
            t4=toc;
            Running_time(4,r)=Running_time(4,r)+t4;
            abs_err_GBa100=[abs_err_GBa100 abs(M_pre(id_test)-M_scores_test(id_test))'];


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
            t5=toc;
            Running_time(5,r)=Running_time(5,r)+t5;
            abs_err_Ori=[abs_err_Ori abs(M_pre(id_test)-M_scores_test(id_test))'];


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
            t6=toc;
            Running_time(6,r)=Running_time(6,r)+t6;
            abs_err_Prop=[abs_err_Prop abs(M_pre(id_test)-M_scores_test(id_test))'];
        end
        Running_time(:,r)=Running_time(:,r)/5;
        MAE(1,r)=mean(abs_err_GBa10);MAE(2,r)=mean(abs_err_GBa20);MAE(3,r)=mean(abs_err_GBa50);
        MAE(4,r)=mean(abs_err_GBa100);MAE(5,r)=mean(abs_err_Ori);MAE(6,r)=mean(abs_err_Prop);
    end

    save('results\Netflix_user_first_Running_time_user_based.mat','Running_time')
    save('results\Netflix_user_first_MAE_user_based.mat','MAE')

else
    load('results\Netflix_user_first_Running_time_user_based.mat')
    load('results\Netflix_user_first_MAE_user_based.mat')
end

figure(1)
plot(ratio,MAE(1,:),'-s',ratio,MAE(2,:),'-*',ratio,MAE(3,:),'-x',ratio,MAE(4,:),'-d',...
    ratio,MAE(5,:),'-o',ratio,MAE(6,:),'-p','Linewidth',1.5);
xlabel('Ratio','Fontsize',16)
ylabel('\bf{MAE}','Fontsize',16)
%title('User-based','Fontsize',16)
l=legend('\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}','\bf{Prop.}');
set(l,'Fontsize',12);
tick={'1%','1.5%','2%','2.5%','3%','3.5%','4%','4.5%','5%'};
set(gca,'XTickLabel',tick)
saveas(gcf,'results\Netflix_user_first_Com_ratio_MAE_user_based.fig')
saveas(gcf,'results\Netflix_user_first_Com_ratio_MAE_user_based.jpg')
saveas(gcf,'results\Netflix_user_first_Com_ratio_MAE_user_based.png')

figure(2)
semilogy(ratio,Running_time(1,:),'-s',ratio,Running_time(2,:),'-*',ratio,Running_time(3,:),'-x',ratio,Running_time(4,:),'-d',...
    ratio,Running_time(5,:),'-o',ratio,Running_time(6,:),'-p','Linewidth',1.5);
xlabel('Ratio','Fontsize',16)
ylabel('Computation time','Fontsize',16)
%title('User-based','Fontsize',16)
l=legend('\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}','\bf{Prop.}');
set(l,'Fontsize',12);
tick={'1%','1.5%','2%','2.5%','3%','3.5%','4%','4.5%','5%'};
set(gca,'XTickLabel',tick)
saveas(gcf,'results\Netflix_user_first_Com_ratio_Running_time_user_based.fig')
saveas(gcf,'results\Netflix_user_first_Com_ratio_Running_time_user_based.jpg')
saveas(gcf,'results\Netflix_user_first_Com_ratio_Running_time_user_based.png')

%%%%%%%%%%%%%%%%%%%%%       item-based    %%%%%%%%%%%%%%%%%%%%%%%%%
% ratio=0.05:-0.005:0.01;


if run
    %% Initialize the outputs
    Running_time=zeros(6,length(ratio));
    MAE=zeros(6,length(ratio));   %% five methods

    for r=1:length(ratio)
        ratio_r=ratio(r);
        abs_err_GBa10=[];abs_err_GBa20=[];abs_err_GBa50=[];
        abs_err_GBa100=[];abs_err_Ori=[];abs_err_Prop=[];
        for i=6:10
            %         if i<=5
            %            n_U=1000;
            %            m_I=1777;
            %         else
            %            n_U=1500;
            %            m_I=888;
            %          end
            M=ratings{i}.data;  %% The row is item and the column is user
            fprintf('* The number of items and users: %d and %d.\n', size(M,1),size(M,2))
            id_all_known=M>0;

            ntest=sum(id_all_known(:))-fix(size(M,1)*size(M,2)*ratio_r);
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
            t1=toc;
            Running_time(1,r)=Running_time(1,r)+t1;
            abs_err_GBa10=[abs_err_GBa10 abs(M_pre(id_test)-M_scores_test(id_test))'];

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
            t2=toc;
            Running_time(2,r)= Running_time(2,r)+t2;
            abs_err_GBa20=[abs_err_GBa20 abs(M_pre(id_test)-M_scores_test(id_test))'];

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
            t3=toc;
            Running_time(3,r)=Running_time(3,r)+t3;
            abs_err_GBa50=[abs_err_GBa50 abs(M_pre(id_test)-M_scores_test(id_test))'];

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
            t4=toc;
            Running_time(4,r)=Running_time(4,r)+t4;
            abs_err_GBa100=[abs_err_GBa100 abs(M_pre(id_test)-M_scores_test(id_test))'];

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
            t5=toc;
            Running_time(5,r)=Running_time(5,r)+t5;
            abs_err_Ori=[abs_err_Ori abs(M_pre(id_test)-M_scores_test(id_test))'];


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
            t6=toc;
            Running_time(6,r)=Running_time(6,r)+t6;
            abs_err_Prop=[abs_err_Prop abs(M_pre(id_test)-M_scores_test(id_test))'];
        end
        Running_time(:,r)=Running_time(:,r)/5;
        MAE(1,r)=mean(abs_err_GBa10);MAE(2,r)=mean(abs_err_GBa20);MAE(3,r)=mean(abs_err_GBa50);
        MAE(4,r)=mean(abs_err_GBa100);MAE(5,r)=mean(abs_err_Ori);MAE(6,r)=mean(abs_err_Prop);
    end


    save('results\Netflix_user_first_MAE_item_based.mat','MAE')
    save('results\Netflix_user_first_Running_time_item_based.mat','Running_time')

else
    load('results\Netflix_user_first_MAE_item_based.mat')
    load('results\Netflix_user_first_Running_time_item_based.mat')
end


figure(3)
plot(ratio,MAE(1,:),'-s',ratio,MAE(2,:),'-*',ratio,MAE(3,:),'-x',ratio,MAE(4,:),'-d',...
    ratio,MAE(5,:),'-o',ratio,MAE(6,:),'-p','Linewidth',1.5);
xlabel('Ratio','Fontsize',16)
ylabel('\bf{MAE}','Fontsize',16)
%title('Item-based','Fontsize',16)
l=legend('\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}','\bf{Prop.}');
set(l,'Fontsize',12);
tick={'1%','1.5%','2%','2.5%','3%','3.5%','4%','4.5%','5%'};
set(gca,'XTickLabel',tick)
saveas(gcf,'results\Netflix_user_first_Com_ratio_MAE_item_based.fig')
saveas(gcf,'results\Netflix_user_first_Com_ratio_MAE_item_based.jpg')
saveas(gcf,'results\Netflix_user_first_Com_ratio_MAE_item_based.png')

figure(4)
semilogy(ratio,Running_time(1,:),'-s',ratio,Running_time(2,:),'-*',ratio,Running_time(3,:),'-x',ratio,Running_time(4,:),'-d',...
    ratio,Running_time(5,:),'-o',ratio,Running_time(6,:),'-p','Linewidth',1.5);
xlabel('Ratio','Fontsize',16)
ylabel('Computation time','Fontsize',16)
%title('Item-based','Fontsize',16)
l=legend('\bf{GBa10}','\bf{GBa20}','\bf{GBa50}','\bf{GBa100}','\bf{Ori}','\bf{Prop.}');
set(l,'Fontsize',12);
tick={'1%','1.5%','2%','2.5%','3%','3.5%','4%','4.5%','5%'};
set(gca,'XTickLabel',tick)
saveas(gcf,'results\Netflix_user_first_Com_ratio_Running_time_item_based.fig')
saveas(gcf,'results\Netflix_user_first_Com_ratio_Running_time_item_based.jpg')
saveas(gcf,'results\Netflix_user_first_Com_ratio_Running_time_item_based.png')

