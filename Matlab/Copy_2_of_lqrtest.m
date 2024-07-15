clc
clear

state = [
 0.5;-0.5;0;0.001;-0.001;0
].';

[t,y] = ode45(@odefun, [0,10000], state);

figure(1)
hold on
plot(t,y(:,1:3))
grid()

figure(2)
hold on
plot(t,y(:,4:6))
grid()


function dydt = odefun(~,y)
a = 6871;
mu = 3.986*10^5;
n = sqrt(mu/a^3);

A = [
    0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 1;
    3*n^2, 0, 0, 0, 2*n, 0;
    0, 0, 0, -2*n, 0, 0;
    0, 0, -n^2, 0, 0, 0;
];


B=[
    0, 0, 0;
    0, 0, 0;
    0, 0, 0;
    1, 0, 0;
    0, 1, 0;
    0, 0, 1;
];


Q = diag([1,1,1,1,10000000000000,1]);

R = diag([1,1,1]);

[K,~,~] = lqr(A,B,Q,R);

u = -K * y;

if norm(u) > 8*10^-5
    u = u/norm(u)*1*10^-5;
end
dydt = A*y + B*u;
end
