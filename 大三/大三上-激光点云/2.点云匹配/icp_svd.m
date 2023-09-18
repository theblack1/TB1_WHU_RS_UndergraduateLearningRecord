function [P_registered,e1,e2]=icp_sys(P,Q,max_iterations, d0)
tic
NS = createns(Q,'NSMethod','kdtree');
j=0;
d=100;
n=size(P,1);
while d>d0
    j=j+1;
    %fprintf("迭代次数：%d\n",j);
    if j>max_iterations
        break
    end
    %寻找Q的对应点集
    [idx, ~] = knnsearch(NS,P,'k',1);
    Qn= Q(idx,:);
    
    %计算旋转矩阵R和平移矩阵t的最优解，使用svd方法
    centerP=mean(P);    %P点集的质心点
    centerQn=mean(Qn);      %对应点集的质心点
    tempP=P-centerP;        %进行去中心化
    tempQn=Qn-centerQn;
    
     H=tempP'*tempQn;  %得到H矩阵
     [U,~,V]=svd(H);
     R=V*U';
        
    %     T=(centerP-centerMap)';
    T=-R*centerP'+centerQn';   %利用质心点求解T参数
    
    %使用R和T来得到新的点集

    P=(R*P'+T)';       %使用转换参数得到新的点集P
    d=sum(sum((P-Qn).^2,2))/n;	%计算新的点集P到对应点的平均距离
    e1=std(sum((P-Qn).^2,2));
    e2=d;
end
P_registered = P;
toc