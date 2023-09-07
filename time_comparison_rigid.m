ws = [0, 0, 0.1, 1, 2; 0, 0.1, 1, 4, 6; 0, 2, 4, 7, 9];
N = 1;
x = 0*unifrnd(-pi/4,pi/4,6,1);
xs = x;
times_dense = zeros(3,5,100);
times_sparse = zeros(3,5,100);
grad_weights = [1, 10, 20];
sparsitys = [0, 0, 0.44, 0.973, 0.989; 0, 0.0027, 0.38, 0.902, 0.95; 0, 0.279, 0.777, 0.889, 0.94];
t = 0;
for i = 1:3
    grad_weight = grad_weights(i);
    for j = 1:5
        w = ws(i,j);
        path = ['./Modelos/Rigid/RigidTanhL1', num2str(w), 't', num2str(t), 'Grad_W', num2str(grad_weight), '.mat'];
        spar = 0;
        f = load_model_from_py_tanh_rigid(path, spar);
        for h = 1:100
        ini = tic;
        [a,b] = f(x);

        tim = toc(ini);
        times_dense(i,j,h) = tim;
        end
        norm(b)
        
        spar = 1;
        fs = load_model_from_py(path, spar);
        for h = 1:100
        ini = tic;
        [as,bs] = fs(xs);
        tim_sp = toc(ini);
        times_sparse(i,j,h) = tim_sp;
        end  
    end
end
times_dense = mean(times_dense,3);
times_sparse = mean(times_sparse,3);
figure(1)
plot(sparsitys(1,:), times_dense(1,:), 'o-')
hold on
plot(sparsitys(2,:), times_dense(2,:), 'o-')
plot(sparsitys(3,:), times_dense(3,:), 'o-')
xlabel('Sparsity','FontSize',25)
ylabel('Evaluation time','FontSize',25)

plot(sparsitys(1,:), times_sparse(1,:), 'o--')
plot(sparsitys(2,:), times_sparse(2,:), 'o--')
plot(sparsitys(3,:), times_sparse(3,:), 'o--')

xtickformat('%,.1f')
ytickformat('%,.1f')
legend('\mu_{Grad}: 1','\mu_{Grad}: 10','\mu_{Grad}: 20','Sparse \mu_{Grad}: 1','Sparse \mu_{Grad}: 10','Sparse \mu_{Grad}: 20','FontSize',20)
hold off