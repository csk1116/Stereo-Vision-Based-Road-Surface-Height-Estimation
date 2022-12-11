fileID = fopen('pitch imu encoder data test.txt', 'r');  
format = '%f , %f , %f , %f , %d';
sizeX = [5 Inf]; 
X = fscanf(fileID, format, sizeX);
fclose(fileID);
X = X';

fileID = fopen('vision pitch.txt', 'r');  
format = '%f';
sizeB = [1 Inf]; 
B = fscanf(fileID, format, sizeB);
fclose(fileID);
B = B';

count=0;
last_count = 0;
num =1;
A = zeros(755,2);
for i=1:3975
    if (X(i,5) == 1)
         count = count+1;
    end
        
    if (X(i,5) == 1 && count == 1)
        last_count = i; 
    end
    
    if (X(i,5) == 1 && count == 2)
            A(num,1) = X(i,2) - X(last_count,2);
            A(num,2) = X(i,4) - X(last_count,4);
            num = num+1;
            count = count-2;
    end
end

C=[A B];

C = C';
fileID = fopen('imu encoder change test.txt','w');
nbytes = fprintf(fileID,'%f, %f, %f\n',C);
fclose(fileID);
 