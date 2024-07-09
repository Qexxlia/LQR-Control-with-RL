clc
clear

d = deg2rad(5);

state = [
 -d;-d;-d;0;0;0
].';

[t,y] = ode45(@odefun, [0,5], state);

figure(1)
hold on
plot(t,rad2deg(y(:,1:3)+[d,d,d]))
legend()
grid()

figure(2)
hold on
plot(t,y(:,4:6))
legend()
grid()


function dydt = odefun(~,y)

A = [
    0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 1;
    0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0;
    0, 0, 0, 0, 0, 0;
];

Kt = 0.0036;
Kf = 0.1188;
L = 0.1969;
Jr = 0.0552;
Jp = 0.0552;
Jy = 0.1104;


B=[
    0, 0, 0, 0;
    0, 0, 0, 0;
    0, 0, 0, 0;
    -Kt/Jy, -Kt/Jy, Kt/Jy, Kt/Jy;
    L*Kf/Jp,-L*Kf/Jp, 0, 0;
    0, 0, L*Kf/Jr, -L*Kf/Jr;
];

C = [
    1, 0, 0, 0, 0, 0;
    0, 1, 0, 0, 0, 0;
    0, 0, 0, 1, 0, 0;
];

D = [
    0,0,0,0;
    0,0,0,0;
    0,0,0,0;
];

sys = ss(A,B,C,D);
    

Q = diag([241,309.6,500,0.01,43.2,131.6]);

R = 0.015*diag([1 1 1 1]);

[K,~,~] = lqr(sys,Q,R);

u = -K * y;

dydt = A*y + B*u;
end
