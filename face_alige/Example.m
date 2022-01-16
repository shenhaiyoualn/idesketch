% Example 
% CropImage using only two eye locations.
clear;
clc;
close all;

imlist = dir('Photos/*.png');

nim = length(imlist);

EPoints = zeros(2,2,nim);

for i = 1:nim

    fprintf('Processing %d/%d image!\n',i,nim);
    
    [path filename ext] = fileparts(imlist(i).name);
  
    im  = imread(['Photos/',imlist(i).name]);

    [im,inem] = CropImage2(im,i,nim);
%     EPoints(:,:,i) = inem(:,:);
     
    imwrite(im,['Results/',imlist(i).name],'jpg');
    
end

% save('EPoints.mat','EPoints');







