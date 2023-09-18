fs=20000;wp=5000;ws=3000;wc=5000;rp=3;rs=30;%数字滤波器指标
omega_p=2*fs*tan((wp*2*pi/fs)/2);
omega_s=2*fs*tan((ws*2*pi/fs)/2);
omega_c=2*fs*tan((wc*2*pi/fs)/2);%预畸变
[n,wn]=buttord(omega_p,omega_s,rp,rs,'s');%阶数n与截止频率wn
[b,a]=butter(n,omega_c,'high','s');%原型模拟滤波器系数b，a
[Ha,omega]=freqs(b,a);%求模拟系统频率特性
dbHa=20*log10(abs(Ha)/max(abs(Ha)));
[bd,ad]=bilinear(b,a,fs);%用双线性变换法求数字滤波器系数bd,ad
[H,w]=freqz(bd,ad);%求数字系统频率特性
dbH=20*log10(abs(H)/max(abs(H)));%化为分贝值
figure;
plot(fs*w/2/pi/1000,dbH),grid,%幅值响应
ylabel('幅值（dB）');xlabel('频率（kHz）');
figure;
freqz(bd,ad);%频谱图
figure;
zplane(bd,ad);
axis([-1.1,1.1,-1.1,1.1]);%零极点图
Hz=tf(bd,ad,1/1000,'Variable','z^-1');


