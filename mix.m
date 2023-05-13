function [net ye yv sonuc r2 MAPE] = mix( mixgirdi,mixcikti,training_rate,n1,lrate )

%MIX Summary of this function goes here
%   Detailed explanation goes here

%input: Gerçekleþmiþ veri girdileri
%target: Gerçekleþmiþ veri çýktýlarý
%training_rate: Training için kullanýlacak veri yüzdesi
%n1: 1. hidden nöron sayýsý
%n2: 2. hidden nöron sayýsý
%lrate:Learning rate

%Fonksiyon function ile baþklar,end ile biter.
%Köþeli parantez çýktý deðiþkenleri için kullanýlýr

%Girdiler fonk.un içine yazýlýr

%input deiþkeni NN girdisi
%target gerçekleþen deðerler

%Girdi training ve validation olarak ayrýlýr
%Çýktý da training ve validation olarak ayrýlýr
%NN oluþtur 
%Trainig girdileri ve training çýktýlarý ile að eðitilir

%  ----Eðitilmiþ Að----

%Aðdan çýkan Validation girdi ve çýktýlarý ile 
 %gerçek çýktýlar karþýlaþtýrýlýr
 
 %  %70 training %30 validation
 
 %sütun yani girilmiþ -girilecek input hesap- deðiþkeni
 
 noofdata = size(mixgirdi,1);
 
 % ntd: number of training data
 %round fonksiyonu yuvarlar
 
 ntd=round(noofdata*training_rate);
 
%Girdi training ve validation olarak ayrýlýr
%Çýktý da training ve validation olarak ayrýlýr

%xt: training girdi
%xv: validation girdi

xt=mixgirdi(1:ntd,:);
xv=mixgirdi(ntd+1:end,:);


%Gerçekleþen sonuçta training ve validation olarak ayrýlýr

%yt: training için gerçekleþen hedef
%yv: validation için gerçekleþen hedef

yt=mixcikti(1:ntd,:);
yv=mixcikti(ntd+1:end,:);

% transpoz [0-1]

xt=xt';
xv=xv';

yt=yt';
yv=yv';

%Girdilerin 0-1 arasý normalizasyon
% xtn : Training için normalize girdi verisi 
% xvn : Validation için normalize girdi verisi

xtn = mapminmax (xt);
xvn = mapminmax (xv);

% Training output'u normalize
% ps normalizasyonun nasýl yapýldýðý ile ilgil bilgiyi saklar

[ytn, ps] = mapminmax (yt);

%net  : að objesi
%newff: feed forward
%{girdi çýktý hidden nöron sayýsý transfer fonk eðitim algoritmasý}
%  {} :Sigmoid fonksiyonu

net=newff(xtn,ytn,n1,{},'trainbr');
rand('state',55);   
net.trainParam.lr=lrate;
net.trainParam.epochs=100000;
net.trainParam.goal=0;
net.trainParam.show=NaN;

%aðýn eðitilmesi

net=train(net,xtn,ytn);

% yen : normalize halde validation çýktýsý

yen=sim(net,xvn);

% yvn : normalize halde validation çýktýsý deðeri yok
% yv  : normalize olmayan gerçek validation
% yen : normalize halde validation çýktýsý deðeri var
% ye  : normalize olmayan validation çýktýsý

ye=mapminmax('reverse', yen ,ps);
ye=ye'; %YSA çýktý
yv=yv'; %gerçek çýktý

MAPE = mean((abs(ye-yv))./yv);

%SStotal = Gerçekleþen Sapma Toplamý
%SSerror = Hatanýn Sapma Toplamý Hata=gerçekleþen-tahmin

sstotal= sum((yv-mean(yv)).^2);
sserror= sum((ye-yv).^2);
r2=1-sserror/sstotal;
sonuc=[ye,yv];

end


