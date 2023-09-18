I=imread('park.png');
a1=imcrop(I);
imwrite(a1,'park moban.jpg','jpg'); %构建一个模板
M=imread('park moban.jpg');%读取模板图像
I1=rgb2gray(I);%将原图灰度化
M1=rgb2gray(M);%将模板图灰度化
[m0,n0]=size(M1);
[m,n]=size(I1);
imshow(I);%显示原图像
hold on;

for i=1:m-m0
for j=1:n-n0
temp_picture=imcrop(I1,[j,i,n0-1,m0-1]);
r=corr2(temp_picture,M1);%取得相关系数
if r>0.95 %规定值为0.95   
%下面用plot函数在原图的坐标系上画出匹配区域
plot(j:j+n0,i,'b');
plot(j:j+n0,i+m0,'b');
plot(j,i:i+m0,'b');
plot(j+n0,i:i+m0,'b');
plot([j,j+n0],[i,i]);
plot([j+n0,j+n0],[i,i+m0]);
plot([j,j+n0],[i+m0,i+m0]);
plot([j,j],[i,i+m0]);
end
end
end