%%�����ֱ��������ȡK��������Ϊ���Լ�
%UI-----��ʼ���ֱ�
%K-----���Լ������ָ���
%testset---K*3�ľ���ÿһ����һ���û���һ����Ŀ�Ĵ�֣�
%          ��һ�б�ʾ��Ŀ���ڶ��б�ʾ�û���������Ϊ��Ӧ���
%rUI----���������ԵĴ�ֲ�����ѵ���������ֱ��н���Ӧ����0

function [testset,rUI]=GetTestSet1(UI,K)
testr=[];
testi=[];
[U,I]=size(UI);
[i,j,v]=find(UI);
ijv=[i,j,v];
k=numel(i);
if k>=K
    rUI=UI;
    rk=randperm(k);
    items=rk(1:K);
    testset=ijv(items,:);
    rUI(testset(:,1)+(testset(:,2)-1)*U)=0;
else
    'K must be less than ratings;'
end
% for k=1:K
%     x=UI(u,:);
%     [temp,ix]=find(x);
%     iix=randperm(length(ix));
%     testr=[testr;x(ix(iix(1:k)))];
%     testi=[testi;ix(iix(1:k))];
%     UI(u,ix(iix(1:k)))=0;
% end