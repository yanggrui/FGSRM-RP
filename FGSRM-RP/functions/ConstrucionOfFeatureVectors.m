function Feature_vectors=ConstrucionOfFeatureVectors(M_scores)
%% Construction of the Feature Vectors

[n,~]=size(M_scores); % n users
Feature_vectors=[];

for u=1:n
    Feature_vector_u=[];
    Score_u=M_scores(u,:);
    Ratings_item_u=find(Score_u); % the item set that user u has give a score
    if isempty(Ratings_item_u)
         Feature_vector_u=zeros(1,8);
    else
       v_u=Score_u(Ratings_item_u)';
       %% The Score matrix for all items of Ratings_item_u
       Score_M=M_scores(:,Ratings_item_u);
       %% the number of all user ratings for each item of of Ratings_item_u
       label_Score_M=Score_M>0;
       num_label=sum(label_Score_M);
       v_u_hat=(sum(Score_M)./num_label)';
       Feature_vector_u=[mean(v_u-v_u_hat),std(v_u-v_u_hat,1)];
       p_u=[sum(v_u==1),sum(v_u==2),sum(v_u==3),...
        sum(v_u==4),sum(v_u==5)]/length(v_u);
    
       mu_u=length(v_u)/max(sum(M_scores>0,2));
       
       Feature_vector_u=[Feature_vector_u,p_u,mu_u]; 
    end
    
 
   
   
%    Feature_vector_u=Feature_vector_u/norm(Feature_vector_u); %% normalization
   
   Feature_vectors=[Feature_vectors Feature_vector_u'];
   
end

end