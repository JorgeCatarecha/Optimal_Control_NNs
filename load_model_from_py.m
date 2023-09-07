function [fun] = load_model_from_py(path, sparse_bool)
data = load(path);

% Access the loaded weights using the variable names
if sparse_bool
    inp_weight = sparse(data.inp_weight);
    inp_bias = sparse(data.inp_bias);
    hidden_0_weight = sparse(data.hidden_0_weight);
    hidden_0_bias = sparse(data.hidden_0_bias);
    hidden_1_weight = sparse(data.hidden_1_weight);
    hidden_1_bias = sparse(data.hidden_1_bias);
    hidden_2_weight = sparse(data.hidden_2_weight);
    hidden_2_bias = sparse(data.hidden_2_bias);
    output_weight = sparse(data.output_weight);
    output_bias = sparse(data.output_bias);
else
    % Access the loaded weights using the variable names
    inp_weight = data.inp_weight;
    inp_bias = data.inp_bias;
    hidden_0_weight = data.hidden_0_weight;
    hidden_0_bias = data.hidden_0_bias;
    hidden_1_weight = data.hidden_1_weight;
    hidden_1_bias = data.hidden_1_bias;
    hidden_2_weight = data.hidden_2_weight;
    hidden_2_bias = data.hidden_2_bias;
    output_weight = data.output_weight;
    output_bias = data.output_bias;
end
fun = @(x) f(x);
function [out, grad] = f(inp)
    x = inp;
    z1 = inp_weight*x+inp_bias';
    x = max(z1,0);
    z2 = hidden_0_weight*x+hidden_0_bias';
    x = max(z2,0);
    z3 = hidden_1_weight*x+hidden_1_bias';
    x = max(z3,0);
    z4 = hidden_2_weight*x+hidden_2_bias';
    x = max(z4,0);
    out = output_weight*x+output_bias';
    grad = repmat(output_weight',1,size(inp,2));
    grad(z4<=0) = 0;
    grad = hidden_2_weight' * grad;
    grad(z3<=0) = 0;
    grad = hidden_1_weight' * grad;
    grad(z2<=0) = 0;
    grad = hidden_0_weight' * grad;
    grad(z1<=0) = 0;
    grad = inp_weight' * grad;
end
end