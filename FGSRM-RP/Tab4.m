%% Tab4: the results is took from the results of running the functions Fig6_user_first and Fig6_item_first
%% rerun the functions Fig6_user_first and Fig6_item_first to obtain new results

load('results\Netflix_user_first_MAE_user_based.mat')
MAE=MAE(:,1:2:9); %% select the results of Ratio=1%,2%,3%,4% and 5%
disp('the user-based prediction results in user-first data:')
MAE=MAE'
clear MAE

load('results\Netflix_user_first_MAE_item_based.mat')
MAE=MAE(:,1:2:9); %% select the results of Ratio=1%,2%,3%,4% and 5%
disp('the item-based prediction results in user-first data:')
MAE=MAE'
clear MAE

load('results\Netflix_item_first_MAE_user_based.mat')
MAE=MAE(:,1:2:9); %% select the results of Ratio=1%,2%,3%,4% and 5%
disp('the user-based prediction results in item-first data:')
MAE=MAE'
clear MAE


load('results\Netflix_item_first_MAE_item_based.mat')
MAE=MAE(:,1:2:9); %% select the results of Ratio=1%,2%,3%,4% and 5%
disp('the item-based prediction results in item-first data:')
MAE=MAE'
clear MAE
