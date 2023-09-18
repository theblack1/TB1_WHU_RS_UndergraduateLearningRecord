wpl=0.3*pi;%定义基本参数
wph=0.7*pi;
wsl=0.4*pi;
wsh=0.6*pi;
wcl=(wpl+wsl)/2;
wch=(wph+wsh)/2;
N=ceil(3.1*2*pi/(wsl-wpl));%计算滤波器长度
n=0:N-1;
alpha=ceil((N-1)/2);
m=n-alpha+eps;
hd=(sin(wcl*m)+sin(pi*m)-sin(wch*m))./(pi*m);%理想单位脉冲响应
w=0.5*(1-cos(2*pi*n/(N-1)));
h=hd.*w;%用hanning窗函数截取
figure(1);
subplot(221);
stem(n,hd);
title('理想单位脉冲响应');
subplot(222);
stem(n,h);
title('用hannig窗函数截取的单位脉冲响应');
[H,F]=freqz(h,1);
subplot(223);
plot(F,20*log10(abs(H)));
xlabel('频率/(rad/s)');
ylabel('幅值/(dB)');
Hz=tf(h,1,1/1000,'Variable','z^-1');%求出系统函数表达式
figure(2);
zplane(h,1);%零极点图
figure(3);
fs=1000;                
t=n/fs;                  
x=sin(2*pi*100*t)+sin(2*pi*250*t);               
plot(n,x);  
xlabel('时间');
ylabel('幅值');
title('原始信号和滤波后信号');
hold on;
y=filter(h, 1, x);
plot(n, y, 'r');
legend('原始信号','滤波后信号');
grid on;