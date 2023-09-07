function [V, Vx, sol_final] = PMP_Solver_Marching(f,g,L,F,Fx,x0,t0,T,u0,guess0,N,u_optimal)
    % Same as Lobatto but using marching time adaptation
    % N is number of divisions of interval [t0, T]
    guess = @(t) guess0(t0+(t-t0)*N);
    if ~exist('u_optimal','var')
    % parameter does exist, we use it
        u_optimal = 0;
    end
    for i = 1:N
      tf = t0+(T-t0)/N*i;
      [v, vx, sol_final_i] = PMP_Solver_Lobatto(f,g,L,F,Fx,x0,t0,tf,u0,guess,u_optimal);
      guess = @(t) deval(sol_final_i,t0+(t-t0)*i/(i+1)*(1-1e-14));
    end
    V = v; Vx = vx; sol_final = sol_final_i;
end