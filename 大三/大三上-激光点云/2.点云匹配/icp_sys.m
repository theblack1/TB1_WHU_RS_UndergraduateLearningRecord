function [P_registered,e1,e2]=icp_sys(P,X,max_iterations, d0)
tic
NS = createns(X,'NSMethod','kdtree');
j=0;
d=100;
n=size(P,1);
P_ = P;
while d>d0
    P = P_;
    j=j+1;
    %fprintf("迭代次数：%d\n",j);
    if j>max_iterations
        break
    end
    %寻找Q的对应点集
    [idx, ~] = knnsearch(NS,P,'k',1);
    Qn= X(idx,:);
    
    %计算旋转矩阵R和平移矩阵t的最优解，使用四元数
    centerP=mean(P);    %P点集的质心点
    centerQn=mean(Qn);      %对应点集的质心点
    P1=P-centerP;        %进行去中心化
    X1=Qn-centerQn;
    
    %计算点集协方差
    sigma=P1'*X1/(length(X1));
    sigma_mi = sigma - sigma';
    M=sigma+sigma'-trace(sigma)*[1,0,0;0,1,0;0,0,1];
    
    %由协方差构造4*4对称矩阵
    Q=[trace(sigma) sigma_mi(2,3) sigma_mi(3,1) sigma_mi(1,2);
       sigma_mi(2,3) M(1,1) M(1,2) M(1,3);
       sigma_mi(3,1) M(2,1) M(2,2) M(2,3);
       sigma_mi(1,2) M(3,1) M(3,2) M(3,3)];

    %计算特征值与特征向量
    [x,y] = eig(Q);
    e = diag(y);

    %计算最大特征值对应的特征向量
    lamda=max(e);
    for i=1:length(Q)
        if lamda==e(i)
            break;
        end
    end
    q=x(:,i);

    q0=q(1);q1=q(2);q2=q(3);q3=q(4);

    %由四元数构造旋转矩阵
    RR=[q0^2+q1^2-q2^2-q3^2 ,2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2);
       2*(q1*q2+q0*q3), q0^2-q1^2+q2^2-q3^2, 2*(q2*q3-q0*q1);
       2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0^2-q1^2-q2^2+q3^2];

    %计算平移向量
    qr=centerQn-centerP*RR';

    %验证旋转矩阵与平移向量正确性
    P_ = P*RR'+qr;
    
    d=sum(sum((P-Qn).^2,2))/n;	%计算新的点集P到对应点的平均距离
    e1=std(sum((P-Qn).^2,2));
    e2=d;
end
P_registered = P;
toc