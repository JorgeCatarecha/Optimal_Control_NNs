function [V, Vx, sol_final] = PMP_Solver_Shooting(f,g,L,F,Fx,x0,t0,T,u0,lambd_0)
    n = length(x0);
    H = @(t,x,lambd,u) L(t,x,u) + lambd'*f(t,x,u);
    per = [t0,T];
    opcs = optimset('Display','iter','TolFun',1e-6); 
    opcs2 = odeset('RelTol',1e-8,'AbsTol',1e-8);
    lambd0 = fsolve(@minf,lambd_0,opcs); 
    w0_optimal = [x0, lambd0];
    sol_final = ode45(@odefun,per,w0_optimal,opcs2);
    xT = deval(sol_final,T,1:n);
    V = F(xT) + integral(@L_opt,t0,T,'arrayvalued',true);
    Vx = deval(sol_final,t0,n+1:2*n);
    function [l] = L_opt(t)
        wt = deval(sol_final,t);
        l = L(t,wt(1:n),u(t,wt(1:n),wt(n+1:2*n)));
    end
    function [dif] = minf(lambd0)
        n = length(lambd0);
        % Shooting method
        w0 = [x0; lambd0];
        sol = ode45(@odefun,per,w0,opcs2);
        wT = deval(sol,T);
        xT = wT(1:n);
        lambdT = wT(n+1:2*n);
        dif = lambdT - Fx(xT);
    end

    function [dw] = odefun(t,w)
        n = length(w)/2;
        x = w(1:n);
        lambd = w(n+1:2*n);
        u_opt = u(t,x,lambd); 
        dx = f(t,x,u_opt);
        dlambd = g(t,x,lambd,u_opt);
        dw = [dx; dlambd];
    end

    function [u_opt] = u(t,x,lambd)
        Haux = @(u) H(t,x,lambd,u);
        opcions_f = optimset('TolFun', 1e-8, 'TolX', 1e-8);
        u_opt = fminsearch(Haux,u0,opcions_f);
    end
end
