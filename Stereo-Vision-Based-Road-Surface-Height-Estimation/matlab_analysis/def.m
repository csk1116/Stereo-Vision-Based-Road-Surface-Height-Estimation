
I1 = imread('left_road55.png');
I2 = imread('right_road55.png');
I1 = imcrop(I1,[57, 82, 1148, 550]);
I2 = imcrop(I2,[57, 82, 1148, 550]);


%title('Red-cyan composite view of the stereo images');

I11=histeq(rgb2gray(I1));
I22=histeq(rgb2gray(I2));

%I11= rgb2gray(I1);
%I22= rgb2gray(I2);

imtool(I11);
imtool(I22);

disparityRange = [10 58];
disparityMap = disparity(I11,I22, 'BlockSize',15 , 'DisparityRange', disparityRange);

figure 
imshow(disparityMap, [10 35] );
%imtool(disparityMap);
title('Disparity Map');
%colormap(gca,jet) 
colorbar


