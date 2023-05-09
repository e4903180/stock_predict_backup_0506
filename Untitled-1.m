MATLAB code:
% Part 1
load PANWdata.mat
for i=1:415
 x(i) = i;
end
p1 = polyfit(x', PANWdata(1:415), 3);
xx = (1:1:505)';
pp1 = polyval(p1, xx);
plot(xx, NFLXdata)
hold
plot(xx, pp1)
xlabel('time(days)')
ylabel('stock price')
title('least squares vs. stock price')
print -dpsc2 -r600 PANWdata.ps
hold
%% difference between data and trend for the first 415 days diff=PANWdata(1:415)-
pp1(1:415);
diff = PANWdata(1:415)-pp1(1:415);
figure
plot(x', diff)
xlabel('time(days)')
ylabel('stock price')
title('difference between least squares and stock price first 415 days')
print -dpsc2 -r600 -append PANWdata1.ps
%% for cleaning the data
% for i=1:415
% if Y(i) > .1*Y(i);
% Y(i) = 0;
% end
% end
Y=fft(diff);
figure
plot(abs(Y))
xlabel('frequency(rad/sec)')
ylabel('fft')
title('abs(fft) of difference between least squares and stock price first 415 days')
print -dpsc2 -r600 -append PANWdata2.ps
%% for cleaning data
% for i=1:415
% if Y(i) > Y(5);
% Y(i) = 0;
% end
% end
PP=ifft(Y);
plot(PP)
figure
plot(PP-diff)
xlabel('time(days)')
ylabel('stock price')
title('ifft of cleaned difference and original difference')
print -dpsc2 -r600 -append PANWdata3.ps
%interpolate the difference for 90days
for n=416:505
YY(n)=0;
 for k=1:415
 a(k)=real(Y(k));
 b(k)=-imag(Y(k));
 omk=2*pi*(k-1)/415; YY(n)=YY(n)+a(k)*cos(omk*(n-1))+b(k)*sin(omk*(n-1));
 end
YY(n)=-YY(n)/415;
end
% create one array for the original and interpolated difference
% between the least squares curve and the stock price. for i=1:415
% figure
% plot(YY)
i=1:415;
YY(i)=diff(i);
%end
%total price of stock from the least squares curve and cleaned fft
%for 505 days
tot= pp1+YY';
%plot the stock price (green) for 505 day vs the theoretical curve(red)
plot(xx,PANWdata,'b');
hold on
plot(xx,tot,'r')
hold off
xlabel('time(days)')
ylabel(' price')
title('Stock price. Actual vs. theoretical 505 days')
print -dpsc2 -r600 -append PANWdata4.ps