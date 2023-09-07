ts = linspace(0,3,11);
ts = ts(1:10);
ws = [0, 2, 6, 10];
N = 1;
x = unifrnd(-3,3,2,1);
xs = sparse(x);
times_dense = zeros(10,length(ws));
times_sparse = zeros(10,length(ws));
veces = 100;
for i = 1:10
    for j = 1:length(ws)
        t = ts(i); w = ws(j);
        path = ['./Modelos/SmallBigL1', num2str(w), 't', num2str(t,'%.1f'), '.mat'];
        
        spar = 0;
        f = load_model_from_py(path, spar);
        ini = tic;
        for k = 1:veces
        [a,b] = f(x);
        end
        tim = toc(ini)/veces;
        times_dense(i,j) = tim;
        
        spar = 1;
        fs = load_model_from_py(path, spar);
        ini = tic;
        for k = 1:veces
        [as,bs] = fs(xs);
        end
        tim_sp = toc(ini)/veces;
        times_sparse(i,j) = tim_sp;
    end
end

figure(10)
plot(ts, times_dense, 'o-')

hold on
plot(ts, times_sparse, 'o--')
xlabel('t','FontSize',30)
ylabel('Evaluation time','FontSize',30)
legend('\mu: 0','\mu: 2','\mu: 6','\mu: 10','Sparse \mu: 0','Sparse \mu: 2','Sparse \mu: 6','Sparse \mu: 10','FontSize',25)
hold off