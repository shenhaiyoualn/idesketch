function [im,inem] = CropImage2(nim,num_i,num_nim)

close all;
figure;
warning off;
% imshow(nim);
imshow(nim,[]);
hold on;
set(gcf,'outerposition',get(0,'screensize'));

for i = 1:2
    switch i 
        case 1
            title(['Left Click Left Eye Center ',num2str(num_i),'/',num2str(num_nim)]);
        case 2
            title(['Left Click Right Eye Center ',num2str(num_i),'/',num2str(num_nim)]);
    end
	temp = ginput(1);
    plot(temp(1),temp(2),'g*');
    inem(i,:) = temp;
end
x1 = inem(1,1);
y1 = inem(1,2);
x2 = inem(2,1);
y2 = inem(2,2);

[height, width, ch] = size(nim);
x0 = width/2;
y0 = height/2;

theta = atan((y2-y1)/(x2-x1));


nim = imcomplement(nim);
im = imrotate(nim,theta*180/pi,'bicubic') ;
im = imcomplement(im);

[height2, width2, ch2] = size(im);
x20 = width2/2;
y20 = height2/2;

deltax = cos(-theta)*(x1-x0)-sin(-theta)*(y1-y0);
deltay = sin(-theta)*(x1-x0)+cos(-theta)*(y1-y0);
x12 = x20+deltax;
y12 = y20+deltay;

deltax = cos(-theta)*(x2-x0)-sin(-theta)*(y2-y0);
deltay = sin(-theta)*(x2-x0)+cos(-theta)*(y2-y0);
x22 = x20+deltax;
y22 = y20+deltay;

% figure,imshow(im);
% hold on;
% plot(x12,y12,'g*'),plot(x22,y22,'g*');

resize = (127-75)/(x22-x12);

im = imresize(im,resize);

x1fin = fix(x12*resize);
y1fin = fix(y12*resize);
x2fin = fix(x22*resize);
y2fin = fix(y22*resize);

% figure,imshow(im);
% hold on;
% plot(x1fin,y1fin,'g*'),plot(x2fin,y2fin,'g*');

[height3, width3, ch] = size(im);
im = im2double(im);
if ch==1
    im(:,:,2) = im(:,:);
    im(:,:,3) = im(:,:,1);
end
finalim = ones(height3+400,width3+500,3);

finalim(201:200+height3,251:250+width3,1) = im(:,:,1);
finalim(201:200+height3,251:250+width3,2) = im(:,:,2);
finalim(201:200+height3,251:250+width3,3) = im(:,:,3);
% figure,imshow(finalim);
% hold on;
% plot(x1fin+250,y1fin+200,'g*'),plot(x2fin+250,y2fin+200,'g*');

% result = zeros(250,200,3);
% result(:,:,1) = finalim(x1fin-74+250:x1fin+125+250,y1fin-124+200:y2fin+125+200,1);
% result(:,:,2) = finalim(x1fin-74+250:x1fin+125+250,y1fin-124+200:y2fin+125+200,2);
% result(:,:,3) = finalim(x1fin-74+250:x1fin+125+250,y1fin-124+200:y2fin+125+200,3);
result(:,:,1) = finalim(y1fin-124+200:y2fin+125+200,x1fin-74+250:x1fin+125+250,1);
result(:,:,2) = finalim(y1fin-124+200:y2fin+125+200,x1fin-74+250:x1fin+125+250,2);
result(:,:,3) = finalim(y1fin-124+200:y2fin+125+200,x1fin-74+250:x1fin+125+250,3);

im = result;


close all;




