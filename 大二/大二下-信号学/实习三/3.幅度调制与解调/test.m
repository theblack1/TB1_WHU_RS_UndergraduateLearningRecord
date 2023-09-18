% 信号幅度的调制与解调
% AM方法调制与解调
clc;
%定义一些初始量
dt=0.001;%时间采样间隔
Fs=1000;
w0=0.5;%信源频率
w1=5;%载波频率
T=4;%信号时长
N=T/dt;%采样点数
t=[0:N-1].*dt;%采样时间序列

s0=cos(2*pi*w0*t);%基波信号
[W0,F0]=T2F(t,s0);%基波信号傅里叶变换
subplot(5,2,1);
plot(t,s0);
xlabel('t');
ylabel('f(t)');
title('基波信号的时域表示');
subplot(5,2,2);
plot(W0,abs(F0));
xlabel('w');
ylabel('F(jw)');
title('基波信号的频域表示');

s1=cos(2*pi*w1*t);%载波信号
[W1,F1]=T2F(t,s1);%载波信号的傅里叶变换
subplot(5,2,3);
plot(t,s1);
xlabel('t');
ylabel('f1(t)');
title('载波信号的时域表示');
subplot(5,2,4);
plot(W1,F1);
xlabel('w');
ylabel('F1(jw)');
title('载波信号的频域表示');

%开始调制
A=3;
s2=(A+s0).*s1;
[W2,F2]=T2F(t,s2);%已调信号的傅里叶变换
subplot(5,2,5);
plot(t,s2);
xlabel('t');
ylabel('f2(t)');
hold on;
plot(t,A+s0,'r--');
title('已调信号的时域表示');
subplot(5,2,6);
plot(W2,F2);
xlabel('w');
ylabel('F2(jw)');
title('已调信号的频域表示');

%相干解调
s3=s2.*s1;
[W3,F3]=T2F(t,s3);%解调后函数的傅里叶变换
subplot(5,2,7);
plot(t,s3);
xlabel('t');
ylabel('f3(t)');
hold on;
plot(t,A+s0,'r--');
title('解调信号的时域表示');
subplot(5,2,8);
plot(W3,F3);
xlabel('w');
ylabel('F2(jw)');
title('解调信号的频域表示');
[W4,F4]=lpf(W3,F3,1);%低通滤波
[W5,F5]=T2F(W4,F4);%低通滤波后的傅里叶变换
subplot(5,2,9);
plot(W4,F4);
title('低通滤波后的时域表示');
subplot(5,2,10);
plot(W5,F5);
title('低通滤波后的频域表示');

%傅里叶变换函数
function [f,sf]= T2F(t,st)
% dt = t(2)-t(1);
T=t(end);
df = 1/T;
N = length(st);
f=-N/2*df : df : N/2 * df-df;
sf = fft(st);
sf = T/N * fftshift(sf);
end


function[t,st]=F2T(f,Sf)
df=f(2)-f(1);
fmax=(f(end)-f(1)+df);
dt=1/fmax;
N=length(f);
t=[0:N-1] * dt;
Sf=fftshift(Sf);
st=fmax * ifft(Sf);
st=real(st);
end

%低通滤波函数
function[t,st]=lpf(f,sf,B)
df=f(2)-f(1);
fN=length(f);
ym=zeros(1,fN);
xm=floor(B/df);
xm_shift=[-xm:xm-1]+floor(fN/2);
ym(xm_shift)=1;
yf=ym.* sf;
[t,st]=F2T(f,yf);
end