function [net ye yv sonuc r2 MAPE] = mix( mixgirdi,mixcikti,training_rate,n1,lrate )

%MIX Summary of this function goes here
%   Detailed explanation goes here

%input: Ger�ekle�mi� veri girdileri
%target: Ger�ekle�mi� veri ��kt�lar�
%training_rate: Training i�in kullan�lacak veri y�zdesi
%n1: 1. hidden n�ron say�s�
%n2: 2. hidden n�ron say�s�
%lrate:Learning rate

%Fonksiyon function ile ba�klar,end ile biter.
%K��eli parantez ��kt� de�i�kenleri i�in kullan�l�r

%Girdiler fonk.un i�ine yaz�l�r

%input dei�keni NN girdisi
%target ger�ekle�en de�erler

%Girdi training ve validation olarak ayr�l�r
%��kt� da training ve validation olarak ayr�l�r
%NN olu�tur 
%Trainig girdileri ve training ��kt�lar� ile a� e�itilir

%  ----E�itilmi� A�----

%A�dan ��kan Validation girdi ve ��kt�lar� ile 
 %ger�ek ��kt�lar kar��la�t�r�l�r
 
 %  %70 training %30 validation
 
 %s�tun yani girilmi� -girilecek input hesap- de�i�keni
 
 noofdata = size(mixgirdi,1);
 
 % ntd: number of training data
 %round fonksiyonu yuvarlar
 
 ntd=round(noofdata*training_rate);
 
%Girdi training ve validation olarak ayr�l�r
%��kt� da training ve validation olarak ayr�l�r

%xt: training girdi
%xv: validation girdi

xt=mixgirdi(1:ntd,:);
xv=mixgirdi(ntd+1:end,:);


%Ger�ekle�en sonu�ta training ve validation olarak ayr�l�r

%yt: training i�in ger�ekle�en hedef
%yv: validation i�in ger�ekle�en hedef

yt=mixcikti(1:ntd,:);
yv=mixcikti(ntd+1:end,:);

% transpoz [0-1]

xt=xt';
xv=xv';

yt=yt';
yv=yv';

%Girdilerin 0-1 aras� normalizasyon
% xtn : Training i�in normalize girdi verisi 
% xvn : Validation i�in normalize girdi verisi

xtn = mapminmax (xt);
xvn = mapminmax (xv);

% Training output'u normalize
% ps normalizasyonun nas�l yap�ld��� ile ilgil bilgiyi saklar

[ytn, ps] = mapminmax (yt);

%net  : a� objesi
%newff: feed forward
%{girdi ��kt� hidden n�ron say�s� transfer fonk e�itim algoritmas�}
%  {} :Sigmoid fonksiyonu

net=newff(xtn,ytn,n1,{},'trainbr');
rand('state',55);   
net.trainParam.lr=lrate;
net.trainParam.epochs=100000;
net.trainParam.goal=0;
net.trainParam.show=NaN;

%a��n e�itilmesi

net=train(net,xtn,ytn);

% yen : normalize halde validation ��kt�s�

yen=sim(net,xvn);

% yvn : normalize halde validation ��kt�s� de�eri yok
% yv  : normalize olmayan ger�ek validation
% yen : normalize halde validation ��kt�s� de�eri var
% ye  : normalize olmayan validation ��kt�s�

ye=mapminmax('reverse', yen ,ps);
ye=ye'; %YSA ��kt�
yv=yv'; %ger�ek ��kt�

MAPE = mean((abs(ye-yv))./yv);

%SStotal = Ger�ekle�en Sapma Toplam�
%SSerror = Hatan�n Sapma Toplam� Hata=ger�ekle�en-tahmin

sstotal= sum((yv-mean(yv)).^2);
sserror= sum((ye-yv).^2);
r2=1-sserror/sstotal;
sonuc=[ye,yv];

end


