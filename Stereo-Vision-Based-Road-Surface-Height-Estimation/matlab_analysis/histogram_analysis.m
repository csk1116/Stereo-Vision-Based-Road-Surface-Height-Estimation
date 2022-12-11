
% ROI
img = imread('left_road31.png');
%imgr = imread('right_road35.png');
%img = imcrop(img,[57, 220, 1148, 412]);
img = imcrop(img,[57, 82, 1148, 550]);
%imgr = imcrop(imgr,[57, 82, 1148, 550]);
%img = [imgl, imgr];

% % ROW RGB & 3 CHANNEL HIST & ENTROPY
% figure;
% imshow(img);
% title('Left RGB road image');
% r = img(:,:,1);
% re = entropy(r);
% %imtool(r);
% g = img(:,:,2);
% ge = entropy(g);
% %imtool(g);
% b = img(:,:,3);
% be = entropy(b);
% %imtool(b);
% figure;
% histogram(r,'BinMethod','integers','FaceColor','r','EdgeAlpha',1,'FaceAlpha',1);
% hold on;
% histogram(g,'BinMethod','integers','FaceColor','g','EdgeAlpha',1,'FaceAlpha',1);
% histogram(b,'BinMethod','integers','FaceColor','b','EdgeAlpha',1,'FaceAlpha',1);
% xlabel('RGB value');
% ylabel('Frequency');
% title('Histogram in RGB color space');
% xlim([0 257]);

% % RGB TO GREY & HIST & ENTROPY
% gs = rgb2gray(img);
% figure;
% imshow(gs);
% title('Left Grayscale road image');
% figure;
% imhist(gs);
% xlabel('Intensity value');
% ylabel('Frequency');
% title('Histogram in grayscale');
% gse = entropy(gs);

% GREY HISTOGRAM EQUALIZATION
% gsqn = histeq(gs);
% figure;
% imshow(gsqn);
% title('Left histogram equalization grayscale road image');
% figure;
% imhist(gsqn);
% title('Histogram of histeq grayscale');
% ylabel('Frequency');
% gsqne = entropy(gsqn);
% 
% % GREY HISTOGRAM EQUALIZATION
% gsq = adapthisteq(gs, 'NumTiles',[5 15],'ClipLimit',0.04);
% figure;
% imshow(gsq);
% title('Left adaptive histogram equalization grayscale road image');
% figure;
% imhist(gsq);
% title('Histogram of adahisteq grayscale');
% ylabel('Frequency');
% gsqe = entropy(gsq);
% 
% 
% 
% RGB TO HSV
% hsv = rgb2hsv(img);
% hsvv = hsv(:,:,3);
% figure;
% imshow(hsvv);
% figure;
% imhist(hsvv);
% hsvve = entropy(hsvv);
% hsvvq =  adapthisteq(hsvv, 'NumTiles',[5 15],'ClipLimit',0.04);
% figure;
% imshow(hsvvq);
% figure;
% imhist(hsvvq);
% hsvvqe = entropy(hsvvq);
% 
% RGB TO YCbCr
ybr = rgb2ycbcr(img);
ybry = ybr(:,:,1);
figure;
imshow(ybry);
figure;
imhist(ybry);
title 'Original Histogram'
xlabel 'Intensity'
ylabel 'Frequency'
ybrye = entropy(ybry);
ybryq =  adapthisteq(ybry, 'NumTiles',[5 15],'ClipLimit',0.04);
figure;
imshow(ybryq);
figure;
imhist(ybryq);
title 'CLAHE Histogram Equalization'
xlabel 'Intensity'
ylabel 'Frequency'
ybryqe = entropy(ybryq);

