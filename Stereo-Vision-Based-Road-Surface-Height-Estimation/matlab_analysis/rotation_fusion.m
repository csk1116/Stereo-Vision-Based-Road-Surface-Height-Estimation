fileID = fopen('pitch frame only 2.txt', 'r');  
format = '%f,%f,%f,%f';
sizeX = [4 Inf]; 
X = fscanf(fileID, format, sizeX);
fclose(fileID);
X = X';

fileID = fopen('vision pitch sum 2.txt', 'r');  
format = '%f';
sizeB = [1 Inf]; 
B = fscanf(fileID, format, sizeB);
fclose(fileID);
B = B';

fusion =0;

for i = 1:1314
    
    if (i==1)
        fusion = fusion + X(i,2);
        X(i,5) = fusion;
    else
        fusion = 0.8*(fusion + B(i-1,1)) + 0.2 * X(i,2);
        X(i,5) = fusion;
    end
end

% count=0;
% last_count = 0;
% num =1;
% A = zeros(755,2);
% for i=1:4270
%     if (X(i,5) == 1)
%          count = count+1;
%     end
%         
%     if (X(i,5) == 1 && count == 1)
%         last_count = i; 
%     end
%     
%     if (X(i,5) == 1 && count == 2)
%             A(num,1) = X(i,2) - X(last_count,2);
%             A(num,2) = X(i,4) - X(last_count,4);
%             num = num+1;
%             count = count-2;
%     end
% end
% 
% C=[A B];
% 
X = X';
fileID = fopen('fusion imu vision pitch 2.txt','w');
nbytes = fprintf(fileID,'%f,%f,%f,%f,%f\n',X);
fclose(fileID);