% Name and functions
name = 'Rigid_Riccati';
opt_ric = odeset('RelTol',1e-8,'AbsTol',1e-8,'Stats','off');
W1 = 1; W2 = 10; W3 = 0.5; W4 = 1; W5 =1;
B = [1 1/20 1/10; 1/15 1 1/10; 1/10 1/15 1];
J = [2 0 0; 0 3 0; 0 0 4];
J_inv = [1/2 0 0; 0 1/3 0; 0 0 1/4];
h = [1 1 1]';
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

L = @(t,x,u) W1/2*(x(1)^2+x(2)^2+x(3)^2) + W2/2*(x(4)^2+x(5)^2+x(6)^2)+W3/2*norm(u)^2;
F = @(x) W4/2*(x(1)^2+x(2)^2+x(3)^2) + W5/2*(x(4)^2+x(5)^2+x(6)^2);
Fx = @(x) [W4*x(1:3);W5*x(4:6)];  % Derivative of F wrt x
T = 20;
u0 = [0 0 0]'; % Initial guess for control when minimising H
vmin = -pi/3; vmax = pi/3; %Range for x0
wmin = -pi/4; wmax = pi/4; %Range for x0
u_optimal = @(t,x,lambd) - (W3^(-1))*transpose(B)*J_inv*[lambd(4);lambd(5);lambd(6)];
t0 = 0;  % t in [t0, T]
%% Riccati
u_fun = - (W3^(-1))*transpose(B)*J_inv*[lambda4; lambda5; lambda6];
auxiliar = [W4*ones(3,1);W5*ones(3,1)];
y0 = reshape(diag(auxiliar),[36,1]);

d_omega_optimal = J_inv * (S * R * h + B * u_fun);
d_total_optimal = [d_v; d_omega_optimal];
H_optimal = 0.5*W1*transpose(v)*v + 0.5*W2*transpose(omega)*omega + 0.5*W3*transpose(u_fun)*u_fun + transpose(lambda) * d_total_optimal;

hess_H_optimal = hessian(H_optimal,[phi theta psi omega1 omega2 omega3 lambda1 lambda2 lambda3 lambda4 lambda5 lambda6]);
dH_dvdv = hess_H_optimal(1:6,1:6);
dH_dldl = hess_H_optimal(7:12,7:12);
dH_dldx = hess_H_optimal(7:12,1:6);
dH_dvdv_func = matlabFunction(dH_dvdv, 'Vars', [phi; theta; psi; omega1; omega2; omega3; lambda1; lambda2; lambda3; lambda4; lambda5; lambda6]);
dH_dvdv_func = @(x,lambd) dH_dvdv_func(x(1),x(2),x(3),x(4),x(5),x(6),lambd(1),lambd(2),lambd(3),lambd(4),lambd(5),lambd(6));
dH_dldl_func = matlabFunction(dH_dldl, 'Vars', [phi; theta; psi; omega1; omega2; omega3; lambda1; lambda2; lambda3; lambda4; lambda5; lambda6]);
dH_dldl_func = @(x,lambd) dH_dldl_func(x(1),x(2),x(3),x(4),x(5),x(6),lambd(1),lambd(2),lambd(3),lambd(4),lambd(5),lambd(6));
dH_dldx_func = matlabFunction(dH_dldx, 'Vars', [phi; theta; psi; omega1; omega2; omega3; lambda1; lambda2; lambda3; lambda4; lambda5; lambda6]);
dH_dldx_func = @(x,lambd) dH_dldx_func(x(1),x(2),x(3),x(4),x(5),x(6),lambd(1),lambd(2),lambd(3),lambd(4),lambd(5),lambd(6));

dH_hess = {dH_dvdv_func, dH_dldl_func, dH_dldx_func};
%%
N = 1;  % Number of times (only initial)
% Data Generation
ts = linspace(t0,T,N+1);
ts = ts(1:N);
M = 10000;  %10000 without Riccati
vs = unifrnd(vmin, vmax, 3, N*M);
ws = unifrnd(wmin, wmax, 3, N*M);
tss = zeros(1,N*M);
D = zeros(N*M,44);
hess_zero = zeros(1, 36);
for i = 1:M*N
    if rem(i,10) == 0
        i
    end
    t = ts(int32(fix((i-1)/M)+1));
    tss(i)=t;
    x = [vs(:,i); ws(:,i)];
    guess = @(t) [x; Fx(x)];
    lastwarn('');
    t = 0;
    tspan = [T t];
    try
    [v, vx, sol_final] = PMP_Solver_Marching(f,g,L,F,Fx,x,t,T,u0,guess,4,u_optimal);
    ode_ric = riccati(dH_hess, sol_final);
    [~, hess_ric] = ode45(ode_ric,tspan,y0,opt_ric);
    hess_final = hess_ric(length(hess_ric),:);
    catch ME
    fprintf(['Error in iteration:',num2str(i),'\n'])
    D(i,:) = [0,0,0,0,0,0,0,hess_zero,1];
    continue;
    end
    [warnMsg, warnId] = lastwarn;
    if isempty(warnMsg)
        D(i,:) = [v, vx', hess_final, 0];
    else
        D(i,:) = [v, vx', hess_zero, 1];
    end
end
D_total = [tss', vs', ws', D];

% Save Dataset
txt = ['.\Datasets\',name,'.csv'];
writematrix(D_total,txt)

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
function [ode_r] = riccati(dH_hess, sol_final)
    function [dw] = ode_riccati(t,w)
        x = deval(sol_final,t,1:6);
        lambd = deval(sol_final,t,7:12);
        w_mat = reshape(w,[6,6])';
        dH_dxdx = dH_hess{1};
        dH_dldl = dH_hess{2};
        dH_dldx = dH_hess{3};
        dw_mat = -dH_dxdx(x,lambd)-w_mat*dH_dldl(x,lambd)*w_mat-w_mat*dH_dldx(x,lambd)-(dH_dldx(x,lambd)')*w_mat;
        dw = reshape(dw_mat',[36,1]);
    end
    ode_r = @(t,w) ode_riccati(t,w);
end