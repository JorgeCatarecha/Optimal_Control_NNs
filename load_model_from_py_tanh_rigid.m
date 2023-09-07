function [fun] = load_model_from_py_tanh_rigid(path, sparse_bool)
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
    x = tanh(z1);
    z2 = hidden_0_weight*x+hidden_0_bias';
    x = tanh(z2);
    z3 = hidden_1_weight*x+hidden_1_bias';
    x = tanh(z3);
    z4 = hidden_2_weight*x+hidden_2_bias';
    x = tanh(z4);
    out = output_weight*x+output_bias';
    
    % Derivative of the output w.r.t. the input (chain rule)
    dout_dz4 = diag(1 - tanh(z4).^2);
    dout_dx = output_weight * dout_dz4;

    dout_dz3 = diag(1 - tanh(z3).^2);
    dout_dx = dout_dx * hidden_2_weight * dout_dz3;

    dout_dz2 = diag(1 - tanh(z2).^2);
    dout_dx = dout_dx * hidden_1_weight * dout_dz2;

    dout_dz1 = diag(1 - tanh(z1).^2);
    dout_dx = dout_dx * hidden_0_weight * dout_dz1;
    
    % The final derivative of the output w.r.t. the input
    grad = dout_dx*inp_weight;
    
%     grad = repmat(output_weight',1,size(inp,2));
%     grad(z4<=0) = 0;
%     grad = hidden_2_weight' * grad;
%     grad(z3<=0) = 0;
%     grad = hidden_1_weight' * grad;
%     grad(z2<=0) = 0;
%     grad = hidden_0_weight' * grad;
%     grad(z1<=0) = 0;
%     grad = inp_weight' * grad;
end
end