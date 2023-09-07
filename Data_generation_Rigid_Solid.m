% Name and functions
name = 'Rigid';
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

N = 1;  % Number of times
% Data Generation
ts = linspace(t0,T,N+1);
ts = ts(1:N);
M = 10;
vs = unifrnd(vmin, vmax, 3, N*M);
ws = unifrnd(wmin, wmax, 3, N*M);
tss = zeros(1,N*M);
D = zeros(N*M,8);
%%
for i = 1:M*N
    if rem(i,10) == 0
        i
    end
    t = ts(int32(fix((i-1)/M)+1));
    tss(i)=t;
    x = [vs(:,i); ws(:,i)];
    guess = @(t) [x; Fx(x)];
    lastwarn('');
    try
    [v, vx, ~] = PMP_Solver_Marching(f,g,L,F,Fx,x,t,T,u0,guess,4,u_optimal);
    catch ME
    fprintf(['Error in iteration:',num2str(i),'\n'])
    D(i,:) = [0,0,0,0,0,0,0,1];
    continue;
    end
    [warnMsg, warnId] = lastwarn;
    if isempty(warnMsg)
        D(i,:) = [v, vx', 0];
    else
        D(i,:) = [v, vx', 1];
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