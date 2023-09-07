%% Trajectory and control comparison VdP
% Exact
global bet
f = @(t,x,u) [x(2);-x(1)+x(2)*(1-x(1).^2)+u];
g = @(t,x,lambd,u) [-2*x(1)+lambd(2)*(1+2*x(1).*x(2));-2*x(2)-lambd(1)-lambd(2)*(1-x(1).^2)];
bet = 0.1;
L = @(t,x,u) sum(x.^2)+bet*u.^2;
F = @(x) 0;
Fx = @(x) 0;  % Derivative of F wrt x
T = 3; t0 = 0;
per = [t0, T];
x0 = [2;-1];
u0 = 0; % Initial guess for control when minimising H
guess2 = @(t) [-3+t; -3+t; 0.5-t; 0.5-t];
lambd_0 = [0;0];
u_optimal = @(t,x,lambd) - lambd(2)/(2*bet);

[v_b, vx_b, sol_final_b] = PMP_Solver_Marching(f,g,L,F,Fx,x0,t0,T,u0,guess2,6,u_optimal);

H = @(t,x,lambd,u) L(t,x,u) + lambd'*f(t,x,u);
points = 100;
ts = linspace(t0,T,points);
xs_b = deval(sol_final_b,ts,[1,2]);
lambds_b = deval(sol_final_b,ts,[3,4]);
% Uncontrol
sol_unc = ode45(@odefun_unc,per,x0);
xs_unc = deval(sol_unc,ts);

% Using NN
tss = linspace(0,3,11);
ts_nn = tss(1:10);
ws = [0, 0.5, 1];
models = {0,0,0};
for j = 1:3
    model = cell(11,1);
    for i = 1:10
        t = ts_nn(i); w = ws(j);
        path = ['./Modelos/L1', num2str(w), 't', num2str(t,'%.1f'), '.mat'];
        spar = 0;
        f = load_model_from_py(path, spar);
        model{i} = f;
    end
    model{11} = @zero_fun;
    models{j} = model;
end

% models_big = {0,0};
% for j = 1:2
%     model = cell(11,1);
%     for i = 1:10
%         t = ts_nn(i); w = ws(j);
%         path = ['./Modelos/BigL1', num2str(w), 't', num2str(t,'%.1f'), '.mat'];
%         spar = 0;
%         f = load_model_from_py(path, spar);
%         model{i} = f;
%     end
%     model{11} = @zero_fun;
%     models_big{j} = model;
% end

% In this case we obtain errors between 4% (for an intermediate
% well-trained) to 21% using the last one. For the first one is 10%,
% probably because it's not well trained.
path_init = './Modelos/L10t0.0.mat';
model_extra = cell(11,1);
for i = 1:10
   f_extra = load_model_from_py(path_init, spar);
   model_extra{i} = f_extra;
end
model_extra{11} = @zero_fun;

odefun_nn_0 = compute_odefun_nn(tss,models{1});
sol_nn_0 = ode45(odefun_nn_0,per,x0);
xs_nn_0 = deval(sol_nn_0,ts);

odefun_nn_1 = compute_odefun_nn(tss,models{2});
sol_nn_1 = ode45(odefun_nn_1,per,x0);
xs_nn_1 = deval(sol_nn_1,ts);

odefun_nn_2 = compute_odefun_nn(tss,models{3});
sol_nn_2 = ode45(odefun_nn_2,per,x0);
xs_nn_2 = deval(sol_nn_2,ts);
% 
% odefun_nn_big_0 = compute_odefun_nn(tss,models_big{1});
% sol_nn_big_0 = ode45(odefun_nn_big_0,per,x0);
% xs_nn_big_0 = deval(sol_nn_big_0,ts);
% 
% odefun_nn_big_1 = compute_odefun_nn(tss,models_big{2});
% sol_nn_big_1 = ode45(odefun_nn_big_1,per,x0);
% xs_nn_big_1 = deval(sol_nn_big_1,ts);

odefun_nn_ini = compute_odefun_nn(tss,model_extra);
sol_nn_ini = ode45(odefun_nn_ini,per,x0);
xs_nn_ini = deval(sol_nn_ini,ts);

% u
us = zeros(length(u0),points);
us_b = zeros(length(u0),points);
us_nn_0 = zeros(length(u0),points);
us_nn_1 = zeros(length(u0),points);
us_nn_2 = zeros(length(u0),points);
us_nn_big_0 = zeros(length(u0),points);
us_nn_big_1 = zeros(length(u0),points);
us_nn_ini = zeros(length(u0),points);
for i = 1:points
   us_b(:,i) = u(ts(i),xs_b(:,i),lambds_b(:,i),u0,H);
   us_nn_0(:,i) = u_int(ts(i), xs_nn_0(:,i),tss,models{1});
   us_nn_1(:,i) = u_int(ts(i), xs_nn_1(:,i),tss,models{2});
   us_nn_2(:,i) = u_int(ts(i), xs_nn_2(:,i),tss,models{3});
%    us_nn_big_0(:,i) = u_int(ts(i), xs_nn_big_0(:,i),tss,models_big{1});
%    us_nn_big_1(:,i) = u_int(ts(i), xs_nn_big_1(:,i),tss,models_big{2});
   us_nn_ini(:,i) = u_int(ts(i), xs_nn_ini(:,i),tss,model_extra);
end

v_unc = trapz(ts, L(0,xs_unc,0));
err_unc = (v_unc-v_b)/v_b*100;
v_nn_0 = trapz(ts, L(0,xs_nn_0,us_nn_0));
err_0 = (v_nn_0-v_b)/v_b*100;
v_nn_1 = trapz(ts, L(0,xs_nn_1,us_nn_1));
err_1 = (v_nn_1-v_b)/v_b*100;
v_nn_2 = trapz(ts, L(0,xs_nn_2,us_nn_2));
err_2 = (v_nn_2-v_b)/v_b*100;
% v_nn_big_0 = trapz(ts, L(0,xs_nn_big_0,us_nn_big_0));
% err_big_0 = (v_nn_big_0-v_b)/v_b*100;
% v_nn_big_1 = trapz(ts, L(0,xs_nn_big_1,us_nn_big_1));
% err_big_1 = (v_nn_big_1-v_b)/v_b*100;
v_nn_ini = trapz(ts, L(0,xs_nn_ini,us_nn_ini));
err_ini = (v_nn_ini-v_b)/v_b*100;
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
plot(ts,xs_nn_1(1,:))
% plot(ts,xs_nn_big_0(1,:))
% plot(ts,xs_nn_big_1(1,:))
plot(ts,xs_nn_ini(1,:))
hold off
legend('Opt','Unc','NN','NN with Penalty', 'NN t=0', 'Location', 'Best')
title('Optimal Trajectory')
xlabel('Time t')
grid on

figure(3)
plot(ts,vecnorm(xs_b))
hold on
plot(ts,vecnorm(xs_unc))
plot(ts,vecnorm(xs_nn_0),'--b')
plot(ts,vecnorm(xs_nn_1),'--k')
plot(ts,vecnorm(xs_nn_2),'--g')
% plot(ts,vecnorm(xs_nn_big_0))
% plot(ts,vecnorm(xs_nn_big_1))
plot(ts,vecnorm(xs_nn_ini))
hold off
%text( 1 , 1 , 'Prueba' )
legend('Optimal','Uncontrolled','NN \mu = 0','NN \mu = 0.5','NN \mu = 1','NN t = 0','Location', 'Best', 'FontSize',20)
%title('Optimal Trajectory Norm','FontSize',14)
xlabel('Time t','FontSize',14)
ylabel('||x||','FontSize',14)
grid on

figure(4)
hold on
plot(ts,us_b,'o-')
plot(ts,us_nn_0,'o--')
plot(ts,us_nn_1,'^--')
plot(ts,us_nn_2,'square--')
% plot(ts,us_nn_big_0,'o')
% plot(ts,us_nn_big_1,'^')
plot(ts,us_nn_ini,'*-')
hold off
grid on
%title('Optimal Control','FontSize',14)
legend('Optimal','NN \mu = 0','NN \mu = 0.5','NN \mu = 1','NN t = 0','Location', 'Best','FontSize',20)
ylabel('Control u','FontSize',14)
xlabel('Time t','FontSize',14)

fprintf('The optimal cost is %.3f. \n',v_b)
fprintf('The uncontrolled cost is %.3f. Error: %.3f%%. \n',v_unc, err_unc)
fprintf('The cost obtain using neural network without penalty is %.3f. Error: %.3f%%. \n',v_nn_0, err_0)
fprintf('The cost obtain using neural network penalty: 0.5 is %.3f. Error: %.3f%%. \n',v_nn_1, err_1)
fprintf('The cost obtain using neural network penalty: 1 is %.3f. Error: %.3f%%. \n',v_nn_2, err_2)
% fprintf('The cost obtain using big neural network without penalty is %.3f. Error: %.3f%%. \n',v_nn_big_0, err_big_0)
% fprintf('The cost obtain using big neural network penalty: 1 is %.3f. Error: %.3f%%. \n',v_nn_big_1, err_big_1)
fprintf('The cost obtain using only the first neural network without penalty is %.3f. Error: %.3f%%. \n',v_nn_ini, err_ini)

function [u_opt] = u(t,x,lambd,u0,H)
    Haux = @(u) H(t,x,lambd,u);
    u_opt = fminsearch(Haux,u0);
end

function [dx] = odefun_unc(~,x)
    dx = [x(2);-x(1)+x(2)*(1-x(1).^2)];
end

function [odefun_nn] = compute_odefun_nn(tss,model)
    function [dx] = aux_fun(t,x)
        dx = [x(2);-x(1)+x(2)*(1-x(1).^2)+u_int(t,x,tss,model)];
    end
    odefun_nn = @(t,x) aux_fun(t,x);
end

function [u_interp] = u_int(t,x,tss,model)
    global bet
    values = zeros(length(x),length(model));
    for k = 1 : length(model)
        [~,result] = model{k}(x);
        values(:,k) = result;
    end
    value = interp1(tss, values', t);
    u_interp =  - value(2)/(2*bet);
end

function [aux, v] = zero_fun(~)
    aux = 0;
    v = [0;0];
end