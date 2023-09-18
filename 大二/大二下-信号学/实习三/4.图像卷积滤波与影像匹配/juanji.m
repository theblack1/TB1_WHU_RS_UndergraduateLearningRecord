%% 图像卷积
clc,clear
I = imread('park.png');
I1=rgb2gray(I);
figure(1);
imshow(I);
[M,N] = size(I1);%获取原始图像大小
I = double(I);
%% conv2函数实现卷积
M1 = 1/9 * [1 1 1;1 1 1;1 1 1];% 3×3 邻域平均线性平滑滤波
I2 = conv2(I1,M1,'same');
I3 = conv2(I1,M1);
figure(2);
subplot(1,3,1);imshow(uint8(I1));title('灰度图像');
subplot(1,3,2);imshow(uint8(I2));title('conv2函数实现卷积，与I大小相同的卷积');
subplot(1,3,3);imshow(uint8(I3));title('conv2函数实现卷积,完整卷积');
