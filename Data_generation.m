% Name and functions
name = 'VanDerPol_Big2';
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

% Data Generation
ts = linspace(t0,T,N+1);
ts = ts(1:N);
M = 1000;
xs1 = unifrnd(xmin, xmax, 1, N*M);
xs2 = unifrnd(xmin, xmax, 1, N*M);
tss = zeros(1,N*M);
D = zeros(N*M,4);
for i = 1:M*N
    if rem(i,10) == 0
        i
    end
    t = ts(int32(fix((i-1)/M)+1));
    tss(i)=t;
    x = [xs1(i); xs2(i)];
    guess = @(t) [x(1)*(1-t/T); x(2)*(1-t/T) ;sign(x(1)*x(2))*(1-t/T); sign(x(1)*x(2))*(1-t/T)];
    lastwarn('');
    try
    [v, vx, ~] = PMP_Solver_Marching(f,g,L,F,Fx,x,t,T,u0,guess,15,u_optimal);
    catch ME
    fprintf(['Error in iteration:',num2str(i),'\n'])
    D(i,:) = [0,0,0,1];
    continue;
    end
    [warnMsg, warnId] = lastwarn;
    if isempty(warnMsg)
        D(i,:) = [v, vx', 0];
    else
        D(i,:) = [v, vx', 1];
    end
end

D_total = [tss', xs1', xs2', D];

% Save Dataset
txt = ['.\Datasets\',name,'.csv'];
writematrix(D_total,txt)