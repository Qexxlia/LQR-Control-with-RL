clc
clear

g=9.81;

A = [
    0, 0, 0,   0, 0, 0,   1, 0, 0,   0, 0, 0;
    0, 0, 0,   0, 0, 0,   0, 1, 0,   0, 0, 0;
    0, 0, 0,   0, 0, 0,   0, 0, 1,   0, 0, 0;
    
    0, 0, 0,   0, 0, 0,   0, 0, 0,   1, 0, 0;
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 1, 0;
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 1;

    0, 0, 0,   0, g, 0,   0, 0, 0,   0, 0, 0;
    0, 0, 0,   -g,0, 0,   0, 0, 0,   0, 0, 0;
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0;

    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0;
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0;
    0, 0, 0,   0, 0, 0,   0, 0, 0,   0, 0, 0;
];


Ixx = 0.11;
Iyy = 0.11;
Izz = 0.04;
m = 1;

B=[
    0,   0,   0,   0;
    0,   0,   0,   0;
    0,   0,   0,   0;
    
    0,   0,   0,   0;
    0,   0,   0,   0;
    0,   0,   0,   0;
    
    0,   0,   0,   0;
    0,   0,   0,   0;
    1/m,  0,   0,   0;
    
    0,   1/Ixx, 0,   0;
    0,   0,   1/Iyy, 0;
    0,   0,   0,   1/Izz;
];

Q = eye(12)*1;

R = eye(4)*1000;

[K,S,P] = lqr(A,B,Q,R);

state = [
 0,0,0,...
 0,pi/8,0,...
 0,0,0,...
 0,0,0   
].';

fprintf("x: %f \ny: %f \nz: %f \nroll: %f \npitch: %f \nyaw: %f \ndx: %f \ndy: %f \ndz: %f \ndroll: %f \ndpitch: %f \ndyaw: %f \n", state)
for i = 1:20
    u = -K * state;
    disp("B*u = ")
    disp(B*u)
    disp("A*state =")
    disp(A*state)
    state = state + A*state + B*u;
    fprintf("x: %f \ny: %f \nz: %f \nroll: %f \npitch: %f \nyaw: %f \ndx: %f \ndy: %f \ndz: %f \ndroll: %f \ndpitch: %f \ndyaw: %f \n", state)
end

