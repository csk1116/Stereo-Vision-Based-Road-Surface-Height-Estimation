fileID = fopen('Per tile analysis k4.txt', 'r');  
format = '%f , %f ;';
sizeX = [2 Inf]; 
X = fscanf(fileID, format, sizeX);
fclose(fileID);
X = X';

% figure;
% plot(X(:,1), X(:,2), '.');

meanX = mean(X);
stdX = std(X);
stdmean =0;
countX =0;
%xmean = 0:1:2760; 
for i = 1:523
   if X(i,2) <= (meanX(1,2)+stdX(1,2)) && X(i,2) >= (meanX(1,2)-stdX(1,2))
       stdmean = stdmean + X(i,2);
       countX = countX + 1;
   else
       continue
   end
end

stdmean = stdmean/countX;


opts = statset('Display','final');
[idx,C] = kmeans(X,1,'Distance','cityblock','Replicates',5,'Options',opts);

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3) 
line([1060,1110],[meanX(1,2) meanX(1,2)],'Color', 'k', 'LineStyle', '--','Linewidth', 2)
line([1060,1110],[meanX(1,2)+stdX(1,2) meanX(1,2)+stdX(1,2)],'Color', 'b', 'LineStyle', '--')
line([1060,1110],[meanX(1,2)-stdX(1,2) meanX(1,2)-stdX(1,2)],'Color', 'r', 'LineStyle', '--')
line([1060,1110],[stdmean stdmean],'Color', [0.8 0.3 0.8], 'LineStyle', '--', 'Linewidth', 2)
legend('Cluster 1','Cluster 2','Centroids','Mean','+std', '-std', 'Zscore','Location','NW')

title 'Data analysis within a tile'
xlabel 'Y Down range (mm)'
ylabel 'Z Elevation (mm)'
hold off

