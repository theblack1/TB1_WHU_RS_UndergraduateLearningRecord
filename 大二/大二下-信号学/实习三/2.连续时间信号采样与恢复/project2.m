%以下为信号采样与恢复代码，截止频率wc与采样频率ws大小按需设定即可

wm=1; %信号带宽
wc=wm; %滤波器截止频率
ws=2*wm;%采样频率（临界采样）

%wc=2*wm; %滤波器截止频率
%ws=4*wm;%采样频率（过采样）

%wc=wm; %滤波器截止频率
%ws=wm;%采样频率（欠采样1）

%wc=wm; %滤波器截止频率
%ws=1.5*wm;%采样频率（欠采样2）

%wc=wm; %滤波器截止频率
%ws=1.75*wm;%采样频率（欠采样3）

Ts=2*pi/ws;%采样间隔
n=-100:100; %时域采样点数
nTs=n*Ts; %时域采样点
f=sinc(nTs/pi); %信号f(nTs)的表达式,Sa(t)=sinc(t/pi)
t=-15:0.005:15;
fa=f*Ts*wc/pi*sinc((wc/pi)*(ones(length(nTs),1)*t-nTs'*ones(1,length(t)))); %信号恢复
error=abs(fa-sinc(t/pi)); %求重构信号与原信号的误差
t1=-15:0.5:15;
f1=sinc(t1/pi);
subplot(3,1,1);
stem(t1,f1);
xlabel('kTs'); ylabel('f(kTs)');
title('sa(t)=sinc(t/pi)采样信号');
subplot(3,1,2);
plot(t,fa);
xlabel('t'); ylabel('fa(t)')
title('信号重构sa(t)');
grid on;
subplot(3,1,3);
plot(t,error);
xlabel('t'); ylabel('error(t)');
title('重构信号与原信号的误差error(t)');
