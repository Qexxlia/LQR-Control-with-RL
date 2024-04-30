state = [
      8.205e-2,
        0.816,
        -3.056e-3,
        -1.014e-4,
        -1.912e-4,
        9.993e-4,
]

t = 0


[tout, yout] = ode45(@CW_LQR, 0:0.0001:20, state)

figure(1)
hold on
plot(tout, yout(:, 1), DisplayName='x')
plot(tout, yout(:, 2), DisplayName='y')
plot(tout, yout(:, 3), DisplayName='z')

legend()