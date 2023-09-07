%% Trajectory and control comparison VdP
% Exact
W1 = 1; W2 = 10; W3 = 0.5; W4 = 1; W5 =1;
B = [1 1/20 1/10; 1/15 1 1/10; 1/10 1/15 1];
J = [2 0 0; 0 3 0; 0 0 4];
J_inv = [1/2 0 0; 0 1/3 0; 0 0 1/4];
h = [1 1 1]';
global f
f = @(t,x,u) [E_mat(x)*[x(4) x(5) x(6)]'; J_inv*(S_mat(x)*R_mat(x)*h + B*u)];
% Compute g using symbolic
syms phi theta psi omega1 omega2 omega3 real

v = [phi; theta; psi];
omega = [omega1; omega2; omega3];

% Lagrange multipliers (adjoint variables)
syms lambda1 lambda2 lambda3 lambda4 lambda5 lambda6 real
lambda = [lambda1; lambda2; lambda3; lambda4; lambda5; lambda6];

syms u1 u2 u3 real
u = [u1; u2; u3];

% Expressions for E(v) and S(omega)
E = [1 sin(phi)*tan(theta) cos(phi)*tan(theta);
     0 cos(phi) -sin(phi);
     0 sin(phi)/cos(theta) cos(phi)/cos(theta)];
 
S = [0 omega3 -omega2;
     -omega3 0 omega1;
     omega2 -omega1 0];
 
R = [cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta);
     sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), sin(phi)*cos(theta);
     cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi), cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi), cos(phi)*cos(theta)];

% Dynamics of v (velocity vector)
d_v = E * omega;
% Dynamics of omega (angular velocity vector)
d_omega = J_inv * (S * R * h + B * u);
d_total = [d_v; d_omega];
% Compute the partial derivatives of the Hamiltonian with respect to v
dH_dv = [diff(0.5 * W1 * transpose(v) * v, phi) + diff(transpose(lambda) * d_total, phi);
         diff(0.5 * W1 * transpose(v) * v, theta) + diff(transpose(lambda) * d_total, theta);
         diff(0.5 * W1 * transpose(v) * v, psi) + diff(transpose(lambda) * d_total, psi)];



% Compute the partial derivatives of the Hamiltonian with respect to omega
dH_domega = [diff(0.5 * W2 * transpose(omega) * omega, omega1) + diff(transpose(lambda) * d_total, omega1);
             diff(0.5 * W2 * transpose(omega) * omega, omega2) + diff(transpose(lambda) * d_total, omega2);
             diff(0.5 * W2 * transpose(omega) * omega, omega3) + diff(transpose(lambda) * d_total, omega3)];

% Convert the symbolic derivatives to MATLAB functions
dH_dv_func = matlabFunction(dH_dv, 'Vars', [phi; theta; psi; omega1; omega2; omega3; lambda1; lambda2; lambda3; lambda4; lambda5; lambda6]);
dH_domega_func = matlabFunction(dH_domega, 'Vars', [phi; theta; psi; omega1; omega2; omega3; lambda1; lambda2; lambda3; lambda4; lambda5; lambda6]);

g = @(t,x,lambd,u) -[dH_dv_func(x(1),x(2),x(3),x(4),x(5),x(6),lambd(1),lambd(2),lambd(3),lambd(4),lambd(5),lambd(6));
    dH_domega_func(x(1),x(2),x(3),x(4),x(5),x(6),lambd(1),lambd(2),lambd(3),lambd(4),lambd(5),lambd(6))];

L = @(t,x,u) W1/2*vecnorm(x(1:3,:)).^2 + W2/2*vecnorm(x(4:6,:)).^2+W3/2*vecnorm(u).^2;
F = @(x) W4/2*(x(1)^2+x(2)^2+x(3)^2) + W5/2*(x(4)^2+x(5)^2+x(6)^2);
Fx = @(x) [W4*x(1:3);W5*x(4:6)];  % Derivative of F wrt x
T = 20;
u0 = [0 0 0]'; % Initial guess for control when minimising H
global u_optimal
u_optimal = @(t,x,lambd) - (W3^(-1))*transpose(B)*J_inv*[lambd(4);lambd(5);lambd(6)]*(1-t/T);
t0 = 0;  % t in [t0, T]
x0 = pi*[-1/8, -1/4 1/8, -1/9, -1/4, 1/5]';
guess = @(t) [x0; Fx(x0)];
per = [0,T];
[v_b, vx_b, sol_final_b] = PMP_Solver_Marching(f,g,L,F,Fx,x0,t0,T,u0,guess,3,u_optimal);

H = @(t,x,lambd,u) L(t,x,u) + lambd'*f(t,x,u);
points = 100;
ts = linspace(t0,T,points);
xs_b = deval(sol_final_b,ts,1:6);
lambds_b = deval(sol_final_b,ts,7:12);
% Uncontrol
sol_unc = ode45(@odefun_unc,per,x0);
xs_unc = deval(sol_unc,ts);

% Using NN
tss = linspace(0,20,11);
ts_nn = tss(1:10);
ws = [0, 1];
model = {0,0};
t = 0; w=6;  grad_weight = 10;
path = ['./Modelos/Rigid/RigidTanhL1', num2str(w), 't', num2str(t), 'Grad_W', num2str(grad_weight), '.mat'];
spar = 0;
fun_modelo = load_model_from_py_tanh_rigid(path, spar);
model{1} = fun_modelo;

w = 0;
path = ['./Modelos/Rigid/RigidTanhL1', num2str(w), 't', num2str(t), 'Grad_W', num2str(grad_weight), '.mat'];
spar = 0;
fun_modelo2 = load_model_from_py_tanh_rigid(path, spar);
model{2} = fun_modelo2;

odefun_nn_0 = compute_odefun_nn(tss,model{1});
sol_nn_0 = ode45(odefun_nn_0,per,x0);
xs_nn_0 = deval(sol_nn_0,ts);

odefun_nn_1 = compute_odefun_nn(tss,model{2});
sol_nn_1 = ode45(odefun_nn_1,per,x0);
xs_nn_1 = deval(sol_nn_1,ts);
% 
% odefun_nn_big_0 = compute_odefun_nn(tss,models_big{1});
% sol_nn_big_0 = ode45(odefun_nn_big_0,per,x0);
% xs_nn_big_0 = deval(sol_nn_big_0,ts);
% 
% odefun_nn_big_1 = compute_odefun_nn(tss,models_big{2});
% sol_nn_big_1 = ode45(odefun_nn_big_1,per,x0);
% xs_nn_big_1 = deval(sol_nn_big_1,ts);

% u
us = zeros(length(u0),points);
us_b = zeros(length(u0),points);
us_nn_0 = zeros(length(u0),points);
us_nn_1 = zeros(length(u0),points);

for i = 1:points
   us_b(:,i) = u_optimal(ts(i),xs_b(:,i),lambds_b(:,i));
   modelo = model{1};
   [~,grad] = modelo(xs_nn_0(:,i));
   us_nn_0(:,i) = u_optimal(ts(i),xs_nn_0(:,i),grad);
   us_nn_1(:,i) = u_optimal(ts(i),xs_nn_1(:,i),grad);
%    us_nn_big_0(:,i) = u_int(ts(i), xs_nn_big_0(:,i),tss,models_big{1});
%    us_nn_big_1(:,i) = u_int(ts(i), xs_nn_big_1(:,i),tss,models_big{2});
end

v_unc = trapz(ts, L(0,xs_unc,0));
err_unc = (v_unc-v_b)/v_b*100;
v_nn_0 = trapz(ts, L(0,xs_nn_0,us_nn_0));
err_0 = (v_nn_0-v_b)/v_b*100;
v_nn_1 = trapz(ts, L(0,xs_nn_1,us_nn_1));
err_1 = (v_nn_1-v_b)/v_b*100;
% v_nn_big_0 = trapz(ts, L(0,xs_nn_big_0,us_nn_big_0));
% err_big_0 = (v_nn_big_0-v_b)/v_b*100;
% v_nn_big_1 = trapz(ts, L(0,xs_nn_big_1,us_nn_big_1));
% err_big_1 = (v_nn_big_1-v_b)/v_b*100;

%% Plots
figure(1)
plot(ts,xs_b)
hold on
plot(ts,lambds_b)

hold off
legend('x1','x2','\lambda 1','\lambda 2','Location', 'Best')
title('Optimal Trajectory')
xlabel('Time t')
grid on

figure(2)
plot(ts,xs_b(1,:))
hold on
plot(ts,xs_unc(1,:))
plot(ts,xs_nn_0(1,:))
% plot(ts,xs_nn_big_0(1,:))
% plot(ts,xs_nn_big_1(1,:))
hold off
legend('Opt','Unc','NN','Location', 'Best')
title('Optimal Trajectory')
xlabel('t')
grid on

figure(3)
plot(ts,vecnorm(xs_b))
hold on
plot(ts,vecnorm(xs_unc))
plot(ts,vecnorm(xs_nn_1),'-.k')
plot(ts,vecnorm(xs_nn_0),'--r')
% plot(ts,vecnorm(xs_nn_big_0))
% plot(ts,vecnorm(xs_nn_big_1))
%plot(ts,vecnorm(xs_nn_ini))
hold off
%text( 1 , 1 , 'Prueba' )
legend('Optimal','Uncontrolled','NN Sparsity=0','NN Sparsity=0.95','Location', 'Best', 'FontSize',20)
%title('Optimal Trajectory Norm','FontSize',14)
xlabel('t','FontSize',25)
ylim([0,2.5])
ylabel('||x||','FontSize',25)
grid on

figure(4)
hold on
plot(ts,vecnorm(us_b),'o-')
plot(ts,vecnorm(us_nn_1),'^--')
plot(ts,vecnorm(us_nn_0),'square--')
% plot(ts,us_nn_big_0,'o')
% plot(ts,us_nn_big_1,'^')
%plot(ts,us_nn_ini,'*-')
hold off
grid on
%title('Optimal Control Norm','FontSize',14)
legend('Optimal','NN Sparsity=0','NN Sparsity=0.95','Location', 'Best','FontSize',20)
ylabel('||u||','FontSize',25)
xlabel('t','FontSize',25)

fprintf('The optimal cost is %.3f. \n',v_b)
fprintf('The uncontrolled cost is %.3f. Error: %.3f%%. \n',v_unc, err_unc)
fprintf('The cost obtain using neural network without penalty is %.3f. Error: %.3f%%. \n',v_nn_1, err_1)
fprintf('The cost obtain using neural network with penalty is %.3f. Error: %.3f%%. \n',v_nn_0, err_0)

function [dx] = odefun_unc(t,x)
    global f
    
    dx = f(t,x,[0,0,0]');
end
function [odefun_nn] = compute_odefun_nn(~,model)
    function [dx] = aux_fun(t,x)
        global f 
        W3 = 0.5;
        B = [1 1/20 1/10; 1/15 1 1/10; 1/10 1/15 1];
        J_inv = [1/2 0 0; 0 1/3 0; 0 0 1/4];
        u_prueba = @(t,x,lambd) - (W3^(-1))*transpose(B)*J_inv*[lambd(4);lambd(5);lambd(6)]*(1-t/20);
        [~,grad] = model(x);
        dx = f(t,x,u_prueba(t,x,grad));
    end
    odefun_nn = @(t,x) aux_fun(t,x);
end

function [Ev] = E_mat(x)
    sinphi = sin(x(1));
    cosphi = cos(x(1));
    costhe = cos(x(2));
    tanthe = tan(x(2));
    Ev = [1 sinphi*tanthe cosphi*tanthe; 0 cosphi -sinphi; 0 sinphi/costhe cosphi/costhe]; 
end
function [Rv] = R_mat(x)
    sinphi = sin(x(1));
    cosphi = cos(x(1));
    sinthe = sin(x(2));
    costhe = cos(x(2));
    sinpsi = sin(x(3));
    cospsi = cos(x(3));
    Rv1 = [costhe*cospsi costhe*sinpsi -sinthe];
    Rv2 = [sinphi*sinthe*cospsi-cosphi*sinpsi sinphi*sinthe*sinpsi+cosphi*cospsi costhe*sinphi];
    Rv3 = [cosphi*sinthe*cospsi+sinphi*sinpsi cosphi*sinthe*sinpsi-sinphi*cospsi costhe*cosphi];
    Rv = [Rv1; Rv2; Rv3]; 
end
function [Sw] = S_mat(x)
    Sw = [0 x(6) -x(5); -x(6) 0 x(4); x(5) -x(4) 0]; 
end