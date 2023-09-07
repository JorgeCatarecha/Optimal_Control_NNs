% Name and functions
name = 'VdP_Ric';
opt_ric = odeset('RelTol',1e-5,'AbsTol',1e-5,'Stats','off');
bet = 0.1;
f = @(t,x,u) [x(2);-x(1)+x(2)*(1-x(1).^2)+u];
g = @(t,x,lambd,u) [-2*x(1)+lambd(2)*(1+2*x(1).*x(2));-2*x(2)-lambd(1)-lambd(2)*(1-x(1).^2)];
L = @(t,x,u) sum(x.^2)+bet*u.^2;
F = @(x) 0;
Fx = @(x) 0;  % Derivative of F wrt x
T = 3;
u0 = 0; % Initial guess for control when minimising H
xmin = -3; xmax = 3; %Range for x0
u_optimal = @(t,x,lambd) - lambd(2)/(2*bet);
t0 = 0;  % t in [t0, T]
N = 10;  % Number of times
% With ricatti, 4 dimensions more info
% Data Generation
ts = linspace(t0,T,N+1);
ts = ts(1:N);
M = 200;
xs1 = unifrnd(xmin, xmax, 1, N*M);
xs2 = unifrnd(xmin, xmax, 1, N*M);
tss = zeros(1,N*M);
D = zeros(N*M,8);
y0 = [0 0 0 0]';
% Riccati
dH_dydy = @(x,lambd) [2-2*lambd(2).*x(2), -2*lambd(2).*x(1); -2*lambd(2).*x(1), 2];
dH_dlambddlambd = @(x,lambd) [0 0; 0 -1/(2*bet)];
dH_dydlambd = @(x,lambd) [0 1; -1-2*x(1).*x(2), 1-x(1).^2];
dH_hess = {dH_dydy, dH_dlambddlambd, dH_dydlambd};

for i = 1:M*N
    if rem(i,10) == 0
        i
    end
    t = ts(int32(fix((i-1)/M)+1));
    tss(i)=t;
    x = [xs1(i); xs2(i)];
    %x = [1;1;1;1];
    guess = @(t) [x(1)*(1-t/T); x(2)*(1-t/T) ;sign(x(1)*x(2))*(1-t/T); sign(x(1)*x(2))*(1-t/T)];
    lastwarn('');
    tspan = [T t];
    try
        [v, vx, sol_final] = PMP_Solver_Marching(f,g,L,F,Fx,x,t,T,u0,guess,15,u_optimal);
        ode_ric = riccati(dH_hess, sol_final);
        [~, hess_ric] = ode45(ode_ric,tspan,y0,opt_ric);
        hess_final = hess_ric(length(hess_ric),:);
    catch ME
        fprintf(['Error in iteration:',num2str(i),'\n'])
        D(i,:) = [0,0,0,0,0,0,0,1];
        continue;
    end
    [warnMsg, warnId] = lastwarn;
    if isempty(warnMsg)
        D(i,:) = [v, vx',hess_final,0];
    else
        D(i,:) = [v, vx',0,0,0,0,1];
    end
end

D_total = [tss', xs1', xs2', D];

% Save Dataset
txt = ['.\Datasets\',name,'.csv'];
writematrix(D_total,txt)

function [ode_r] = riccati(dH_hess, sol_final)
    function [dw] = ode_riccati(t,w)
        x = deval(sol_final,t,1:2);
        lambd = deval(sol_final,t,3:4);
        w_mat = [w(1) w(2); w(3) w(4)];
        dH_dxdx = dH_hess{1};
        dH_dldl = dH_hess{2};
        dH_dxdl = dH_hess{3};
        dw_mat = -dH_dxdx(x,lambd)-w_mat*dH_dldl(x,lambd)*w_mat-w_mat*dH_dxdl(x,lambd)-(dH_dxdl(x,lambd)')*w_mat;
        dw = [dw_mat(1,1);dw_mat(1,2);dw_mat(2,1);dw_mat(2,2)];
    end
    ode_r = @(t,w) ode_riccati(t,w);
end