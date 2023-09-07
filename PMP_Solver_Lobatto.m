function [V, Vx, sol_final] = PMP_Solver_Lobatto(f,g,L,F,Fx,x0,t0,T,u0,guess,u_optimal)
    n = length(x0);
    H = @(t,x,lambd,u) L(t,x,u) + (lambd')*f(t,x,u);
    opcs = bvpset('RelTol',1e-8,'AbsTol',1e-8,'Nmax',10000,'Stats','off');
    tmesh = linspace(t0,T);
    solinit = bvpinit(tmesh, guess);
    sol_final = bvp4c(@odefun,@bcfun,solinit,opcs);
    xT = deval(sol_final,T,1:n);
    V = F(xT) + integral(@L_opt,t0,T,'arrayvalued',true);
    Vx = deval(sol_final,t0,n+1:2*n);
    function [l] = L_opt(t)
        wt = deval(sol_final,t);
        l = L(t,wt(1:n),u(t,wt(1:n),wt(n+1:2*n)));
    end
    function [res] = bcfun(w0,wT)
        c1 = w0(1:n)-x0;
        c2 = wT(n+1:2*n)-Fx(wT(1:n));
        res = [c1; c2];
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
        if ~(isequal(u_optimal,0))
            % parameter does exist, we use it
            u_opt = u_optimal(t,x,lambd);
        else
            Haux = @(u) H(t,x,lambd,u);
            opcions_f = optimset('TolFun', 1e-7, 'TolX', 1e-7);
            u_opt = fminsearch(Haux,u0,opcions_f);
        end
    end
end
