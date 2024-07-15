clc
clear

d = deg2rad(5);

state = [
 0;0;0;0;0;0;0;0;0;0;0;0;
].';

[t,X] = ode45(@pos_controller, [0,25], state);


figure(1)
hold on
plot(t,X(:,1:3))
legend()
grid()

figure(2)
hold on
plot(t,X(:,7:9))
legend()
grid()


function dydt = pos_controller(~, X)

    ref = [1,-1,.5];

    X_ang = X(7:12);
    X_pos = X(1:6);

    m = 1;
    g = 9.81;

    A = [
        0, 0, 0, 1, 0, 0;
        0, 0, 0, 0, 1, 0;
        0, 0, 0, 0, 0, 1;
        0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0;
    ];

    B = [
        0, 0, 0;
        0, 0, 0;
        0, 0, 0;
        0, g, 0;
        g, 0, 0;
        0, 0, 1/m;
    ];

    Q =  32*diag([1,1,1,1,1,1]);
    R =  32*diag([1,1,1]);

    [K,~,~] = lqr(A,B,Q,R);

    error = X_pos - [ref,0,0,0].';
    u = -K*error;
    
    dadt = ang_controller(u, X_ang);

    u = X_ang + dadt;

    dxdt = A*X_pos + B*u(1:3);

    dydt = [dxdt; dadt];
end




function dadt = ang_controller(ref, X)
    % Cap
    max = 1;
    ref(ref > max) = max;


    Kt = 0.0036;
    Kf = 0.1188;
    L = 0.1969;
    Jr = 0.0552;
    Jp = 0.0552;
    Jy = 0.1104;


    A = [
        0, 0, 0, 1, 0, 0;
        0, 0, 0, 0, 1, 0;
        0, 0, 0, 0, 0, 1;
        0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0;
        0, 0, 0, 0, 0, 0;
    ];

    B = [
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
        0, 0, 1, 0, 0, 0;
    ];

    Q =  32*diag([1,1,1,1,1,1]);
    R =  32*diag([1,1,1,1]);

    [K,~,~] = lqr(A,B,Q,R);    

    error = X - [ref;0;0;0];

    disp(error)

    % u = -K*error;

    inv([A,B;C,0]) * [0;eye(3)]
    Mu = 
    Mx = 

    u = (Mu-K*Mx)*ref - K*X;

    dadt = A*X + B*u;
end
