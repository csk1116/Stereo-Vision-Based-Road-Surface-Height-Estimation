CX = 551.6546;
CY = 270.6152;
theta = 122*pi/180;
f = 528.7462;
TX = 119.5829;
y = 50;
A = zeros(64,2);
B = zeros(64,2);
HC = 1360;

for d = 1:64
    A(d,1) = 65-d;
    A(d,2) =  TX*(-(y-CY)*sin(theta)+f*cos(theta))/(65-d) + HC ;
end

for d = 1:64
    B(d,1) = 65-d;
    B(d,2) =  TX*((y-CY)*cos(theta)+f*sin(theta))/(65-d);
end

figure;
plot(A(:,1),A(:,2),'r.','MarkerSize',12)

figure;
plot(B(:,1),B(:,2),'r.','MarkerSize',12)

