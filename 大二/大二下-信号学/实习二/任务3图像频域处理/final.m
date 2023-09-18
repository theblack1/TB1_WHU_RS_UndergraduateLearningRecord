A = imread('test.jpg');
I=rgb2gray(A);%图像灰度化处理
imwrite(I,'灰度图像.png');
figure(1);
subplot(421),imshow(A);
title('原图像');
I=im2double(I);
Y=fftshift(fft2(I));%傅里叶变换，直流分量搬移到频谱中心
W=log(abs(Y)+1);
subplot(422), imshow(W,[]); 
title('图像傅里叶变换取对数所得频谱');
imwrite(W,'傅里叶频谱.png');
[M,N]=size(Y);
h=zeros(M,N);%滤波器函数

%低通部分

%理想低通
res=zeros(M,N);%保存结果
M0=round(M/2);
N0=round(N/2);
D0=40;
for i=1:M 
    for j=1:N 
        distance=sqrt((i-M0)^2+(j-N0)^2);
        if distance<=D0
            h(i,j)=1;
        else
            h(i,j)=0;
        end
    end
end
res=Y.*h;
res=real(ifft2(ifftshift(res)));
subplot(423),imshow(res);
title('理想低通滤波所得图像'); 
imwrite(res,'理想低通滤波.png');
subplot(424),imshow(h);
title("理想低通滤波器图象");
imwrite(h,'理想低通滤波器.png');

%高斯低通
%图像中心点
M0=M/2;
N0=N/2;
%截至频率距离圆点的距离，D0表示高斯曲线的扩散程度
D0=40;
for x=1:M
    for y=1:N
        %计算点（x,y）到中心点的距离
        d2=(x-M0)^2+(y-N0)^2;
        %计算高斯滤波器
        h(x,y)=exp(-d2/(2*D0^2));
    end
end
%滤波后结果
res=h.*Y;
res=real(ifft2(ifftshift(res)));
subplot(425),imshow(res);
title('高斯低通滤波所得图像'); 
imwrite(res,'高斯低通滤波.png');
subplot(426),imshow(h);
title("高斯低通滤波器图象");
imwrite(h,'高斯低通滤波器.png');

%巴特沃斯低通
%图像中心点
M0=M/2;
N0=N/2;
D0=40;
%巴特沃斯滤波器的阶数
n_0=2;
for x=1:M
    for y=1:N
        distance=sqrt((x-M0)^2+(y-N0)^2);
        h(x,y)=1/(1+(distance/D0)^(2*n_0));
    end
end
%滤波后结果
res=h.*Y;
res=real(ifft2(ifftshift(res)));
subplot(427),imshow(res);
title('巴特沃斯低通滤波所得图像'); 
imwrite(res,'巴特沃斯低通滤波.png');
subplot(428),imshow(h);
title("巴特沃斯低通滤波器图象");
imwrite(h,'巴特沃斯低通滤波器.png');

%高通部分

figure(2);
subplot(421),imshow(A);
title('原图像');
I=im2double(I);
Y=fftshift(fft2(I));%傅里叶变换，直流分量搬移到频谱中心
subplot(422), imshow(log(abs(Y)+1),[]); 
title('图像傅里叶变换取对数所得频谱');
[M,N]=size(Y);
h=zeros(M,N);%滤波器函数

%理想高通
res=zeros(M,N);%保存结果
M0=round(M/2);
N0=round(N/2);
D0=40;
for i=1:M 
    for j=1:N 
        distance=sqrt((i-M0)^2+(j-N0)^2);
        if distance<D0
            h(i,j)=0;
        else
            h(i,j)=1;
        end
    end
end
res=Y.*h;
res=real(ifft2(ifftshift(res)));
subplot(423),imshow(res);
title('理想高通滤波所得图像'); 
imwrite(res,'理想高通滤波.png');
subplot(424),imshow(h);
title('理想高通滤波器图像'); 
imwrite(h,'理想高通滤波器.png');

%高斯高通
%图像中心点
M0=M/2;
N0=N/2;
%截至频率距离圆点的距离，D0表示高斯曲线的扩散程度
D0=40;
for x=1:M
    for y=1:N
        %计算点（x,y）到中心点的距离
        d2=(x-M0)^2+(y-N0)^2;
        %计算高斯滤波器
        h(x,y)=1-exp(-d2/(2*D0^2));
    end
end
%滤波后结果
res=h.*Y;
res=real(ifft2(ifftshift(res)));
subplot(425),imshow(res);
title('高斯高通滤波所得图像'); 
imwrite(res,'高斯高通滤波.png');
subplot(426),imshow(h);
title("高斯高通滤波器图象");
imwrite(h,'高斯高通滤波器.png');

%巴特沃斯高通
%图像中心点
M0=M/2;
N0=N/2;
D0=40;
%巴特沃斯滤波器的阶数
n_0=2;
for x=1:M
    for y=1:N
        distance=sqrt((x-M0)^2+(y-N0)^2);
        h(x,y)=1/(1+(D0/distance)^(2*n_0));
    end
end
%滤波后结果
res=h.*Y;
res=real(ifft2(ifftshift(res)));
subplot(427),imshow(res);
title('巴特沃斯高通滤波所得图像'); 
imwrite(res,'巴特沃斯高通滤波.png');
subplot(428),imshow(h);
title("巴特沃斯高通滤波器图象");
imwrite(h,'巴特沃斯高通滤波器.png');
