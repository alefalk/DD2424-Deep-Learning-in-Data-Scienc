clear all
clf
clc


% read in and preprocess data

all_training_batches = ["datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_1.mat"; 
                "datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_2.mat";
                "datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_3.mat";
                "datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_4.mat";
                "datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_5.mat"];


[allX, allY, ally] = LoadAll(all_training_batches);
dp = 49000;
trainX = allX(:, 1:dp);
trainY = allY(:, 1:dp);
trainy = ally(1:dp, :);
valX = allX(:, dp+1:end);
valY = allY(:, dp+1:end);
valy = ally(dp+1:end, :);
% [trainX, trainY, trainy] = LoadBatch("datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_1.mat");
% [valX, valY, valy] = LoadBatch("datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_2.mat");
[testX, testY, testy] = LoadBatch("datasets\cifar-10-matlab\cifar-10-batches-mat\test_batch.mat");

mean_trainX = mean(trainX, 2);
std_trainX = std(trainX, 0, 2);

%normalize data
n_trainX = NormalizeData(trainX, mean_trainX, std_trainX);
n_valX = NormalizeData(valX, mean_trainX, std_trainX);
n_testX = NormalizeData(testX, mean_trainX, std_trainX);

%nr_hidden = 2;
input_weight = 50;
%m = [40, 20, 10, 5];

%m(1) = 50;

%subsets
use_sub = false;
if use_sub
    n_trainX = n_trainX(1:200, 1:10);
    n_valX = n_valX(1:200, 1:10);
    n_testX = n_testX(1:200, 1:10);
    trainY = trainY(:, 1:10);
    trainy = trainy(1:10, :);
    valY = valY(:, 1:10);
    valy = valy(1:10, :);
    testy = testy(1:10, :);
end
d = size(n_trainX, 1);
K = size(trainY, 1);
l = 50;
net_params = InitParameters(l, K, d);


lambda = 0.0075;
eta_min = 1e-5;
eta_max = 1e-1;
cycles = 2;
n_batch = 100;
n_s = 5*45000/n_batch;
hyperparams = [n_batch, eta_min, eta_max, n_s, cycles];

l_min = -5;
l_max = -1;
h = 0.0001;
net_params.use_bn = true;


%Coarse-Fine search to find best value of lambda.
%CoarseSearch(l_min, l_max, n_trainX, n_testX, testy, trainY, trainy, n_valX, valY, valy, hyperparams, net_params, 10);

%Testing gradients analytically vs numerically (centered difference
%method)
% [P, x, bn_values] = EvaluateClassifier(n_trainX, net_params, K);
% analytical_Grads = ComputeGradients(n_trainX, trainY, P, net_params, bn_values, x, lambda);
% numerical_Grads = ComputeGradsNumSlow(n_trainX, trainY, net_params, lambda, h);
% 
% for l = 1:length(analytical_Grads.W)
%     rel_grad_error_W = ComputeRelativeError(analytical_Grads.W{l}, numerical_Grads.W{l})
%     rel_grad_error_b = ComputeRelativeError(analytical_Grads.b{l}, numerical_Grads.b{l})
% end
% if net_params.use_bn
%     for l = 1:length(analytical_Grads.gammas)
%         rel_grad_error_gammas = ComputeRelativeError(analytical_Grads.gammas{l}, numerical_Grads.gammas{l})
%         rel_grad_error_betas = ComputeRelativeError(analytical_Grads.betas{l}, numerical_Grads.betas{l})
%     end
% end


[net_params, val_acc] = MBGD(n_trainX, trainY, hyperparams, n_valX, valY, valy, trainy, net_params, lambda, K);
 accuracy_test = ComputeAccuracy(n_testX, testy, net_params, K)


function CoarseSearch(l_min, l_max, n_trainX, n_testX, testy, trainY, trainy, n_valX, valY, valy, hyperparams, net_params, n_lambdas)
    test_accs = zeros(n_lambdas,1);
    lambdas = zeros(n_lambdas,1);
    K = size(trainY, 1);
    for i = 1:n_lambdas
        lambda = RandomSampleLambda(l_min, l_max);
        [net_params, ~] = MBGD(n_trainX, trainY, hyperparams, n_valX, valY, valy, trainy, net_params, lambda, K);
        test_accs(i) = ComputeAccuracy(n_testX, testy, net_params, K);
        lambdas(i) = lambda;
    end
    matlab.io.saveVariablesToScript('coarse_search.m',{'test_accs','lambdas'})
end

function lambda = RandomSampleLambda(l_min, l_max)
    l = l_min + (l_max - l_min) * rand(1, 1);
    lambda = 10^l;
end

function relative_error = ComputeRelativeError(grad_an, grad_num)
    eps = 0.0001;
    relative_error = norm(grad_an - grad_num) / max([eps, norm(grad_an) + norm(grad_num)]);
end

function [X, Y, y] = LoadBatch(filename)
    data_batch = load(filename);
    X = transpose(double(data_batch.data));
    y = double(data_batch.labels) + 1; % from 0-9 to 1-10

    %one hot encoding
    numClasses = 10; 
    numExamples = length(y);
    Y = zeros(numClasses, numExamples);
    for i = 1:numExamples
        Y(y(i), i) = 1;
    end
end

function[X, Y, y] = LoadAll(all_training_batches)
    startIndex = 1;
    endIndex = 10000;
    
    % Loop through each batch file
    for i = 1:5  
        % Load the current batch
        batchData = load(all_training_batches(i)); 
        trainX = batchData.data;
        trainy = batchData.labels + 1;
        
        allX(startIndex:endIndex, :) = trainX;
        ally(startIndex:endIndex, :) = trainy;

        %one hot encoding
        numClasses = 10; 
        numExamples = length(ally);
        allY = zeros(numClasses, numExamples);
        for j = 1:numExamples
            allY(ally(j), j) = 1;
        end

        % Update indices
        startIndex = endIndex + 1;
        endIndex = startIndex + 9999;  % Increment by 10,000 for the next batch
    end 

    X = transpose(double(allX));
    Y = double(allY);
    y = double(ally);
end

function X = NormalizeData(X, mean_trainX, std_trainX) 
    X = X - repmat(mean_trainX, [1, size(X,2)]);
    X = X ./ repmat(std_trainX, [1, size(X,2)]);
end 

%initialization of network using He initialization for weights
function net_params = InitParameters(h_nodes, K, d)
    total_layers = length(h_nodes) + 1;
    W = cell(1, total_layers);
    b = cell(1, total_layers);
    betas = cell(1, total_layers-1);
    gammas = cell(1, total_layers-1);
    
    % input to first hidden
    sigma = sqrt(2/d);
    W{1} = sigma * randn(h_nodes(1), d);
    b{1} = zeros(h_nodes(1), 1);
    betas{1} = zeros(h_nodes(1), 1);
    gammas{1} = ones(h_nodes(1), 1);
    
    %hidden to hidden
    for l=2:length(h_nodes)
        sigma = sqrt(2/h_nodes(l-1));
        W{l} = sigma * randn(h_nodes(l), h_nodes(l-1));
        b{l} = zeros(h_nodes(l), 1);
        betas{l} = zeros(h_nodes(l), 1);
        gammas{l} = ones(h_nodes(l), 1);
    end
    
    %final layer
    sigma = sqrt(2/h_nodes(total_layers-1));
    W{total_layers} = sigma * randn(K, h_nodes(end));
    b{total_layers} = zeros(K, 1);
    
    net_params.W = W;
    net_params.b = b;
    net_params.betas = betas;
    net_params.gammas = gammas;
end

function s_hat = BatchNormalize(s, mu, var)
    s_hat = (diag(var + eps))^(-1/2) * (s - mu);
end

function [P, x, bn_values] = EvaluateClassifier(X, net_params, K, varargin)
     n = size(X, 2);
     P = zeros(K, n);
     S = cell(1, size(net_params.W, 2)-1);
     bn_S = cell(1, size(net_params.W, 2)-1);
     bn_x = cell(1, size(net_params.W, 2)-1);
     x = cell(1, size(net_params.W, 2)-1);
     mu = cell(1, size(net_params.W, 2)-1);
     v = cell(1, size(net_params.W, 2)-1);
     s_hat = cell(1, size(net_params.W, 2)-1);

     for w=1:size(net_params.W, 2)-1
        s = net_params.W{w} * X + net_params.b{w};
        if net_params.use_bn
            mu{w} = mean(s, 2);
            v{w} = transpose(var(s, 0, 2) * (n-1) / n);
            if length(varargin) > 4
                s_hat{w} = BatchNormalize(s, varargin{1}{1}{1}{w}, varargin{1}{1}{2}{w});
            else
                s_hat{w} = BatchNormalize(s, mu{w}, v{w});
            end
            s_tilde = net_params.gammas{w} .* s_hat{w} + net_params.betas{w};
            X = max(0, s_tilde);
            bn_S{w} = s_hat{w};
            bn_x{w} = X;
        else
            X = max(0, s);
        end
        S{w} = s;
        x{w} = X;
     end
     s = net_params.W{end} * x{end} + net_params.b{end};
     for col=1:n
         P(:, col) = softmax(s(:, col));
     end
     bn_values.S = S;
     bn_values.S_hat = bn_S;
     bn_values.x = bn_x;
     bn_values.mu = mu;
     bn_values.v = v;
end

function softmax_result = softmax(s)
    softmax_result = exp(s) ./ sum(exp(s), 1);
end

function G_batch = BatchNormBackPass(G, S, mu, v)
    n = size(G, 2);
    sigma1 = transpose((v + eps) .^ (-0.5));
    sigma2 = transpose((v + eps).^(-1.5));
    G1 = G .* (sigma1*ones(1, n));
    G2 = G .* (sigma2*ones(1, n));
    D = S - mu*ones(1, n);
    c = (G2 .* D) * ones(n, 1);
    G_batch = G1 - ((G1*ones(n, 1))*ones(1, n))/n - (D.*(c*ones(1, n))) / n;
end

function Gradients = ComputeGradients(MB, Y, P, net_params, bn_values, x, lambda)
    n = size(MB, 2);  % Number of samples
    G = -(Y - P);  % Initial gradient from the output
    k = length(net_params.W);
    Gradients.W = cell(numel(net_params.W), 1);
    Gradients.b = cell(numel(net_params.b), 1);
    Gradients.gammas = cell(numel(net_params.gammas), 1);
    Gradients.betas = cell(numel(net_params.betas), 1);

    for w=k:-1:2
        Gradients.W{w} = G*transpose(x{w-1})/n + 2*lambda*net_params.W{w};
        Gradients.b{w} = G*ones(n, 1)/n;
        G = transpose(net_params.W{w})*G;
        G = G.*(x{w-1} > 0);  
        if net_params.use_bn
            Gradients.gammas{w-1} = (G .* bn_values.S_hat{w-1}) * ones(n, 1)/n;
            Gradients.betas{w-1} = G * ones(n, 1) / n;
            G = G .* (net_params.gammas{w-1} * ones(1, n));
            G = BatchNormBackPass(G, bn_values.S{w-1}, bn_values.mu{w-1}, bn_values.v{w-1});
        end
    end
    Gradients.W{1} = G*transpose(MB)/n + 2*lambda*net_params.W{1};
    Gradients.b{1} = G*ones(n, 1)/n;
end


function [J, C_E_Loss] = ComputeCost(X, Y, net_params, lambda, varargin)
    n = size(X, 2);
    r = 0;
    K = size(Y,1);
    C_E_Loss=0;
    [P, ~, ~] = EvaluateClassifier(X, net_params, K, varargin);
    
    for img=1:size(Y,2)
        C_E_Loss = C_E_Loss + -transpose(Y(:, img)) * log(P(:, img));
    end
    
    C_E_Loss = C_E_Loss / n;
    
    for w = 1:length(net_params.W)
        for i = 1:size(net_params.W{w}, 1)
            for j = 1:size(net_params.W{w}, 2)
                r = r + net_params.W{w}(i, j)^2;
            end
        end
    end
    r = lambda*r;
    J = C_E_Loss + r;
end

function eta_t = compute_clr(eta_min, eta_max, n_s, t)
     cycle = floor(1 + t/(2*n_s));
     x = abs(t/n_s - 2*cycle + 1);
     eta_t = eta_min + (eta_max - eta_min) * (max(0, 1-x));
end

function acc = ComputeAccuracy(X, y, net_params, K, varargin)
    n = size(X,2);
    n_errors = 0;
    [P, ~, ~] = EvaluateClassifier(X, net_params, K, true);

    for col = 1:n
        [~, max_p_index] = max(P(:, col));
        if max_p_index ~= y(col)
            n_errors = n_errors + 1;
        end
    end
    acc = (length(y) - n_errors) / length(y);
end


function [mu_movingavg, v_movingavg] = ComputeMovingAverage(mu_movingavg, v_movingavg, bn_values)
    alpha = 0.7;
    for w=1:length(mu_movingavg)
        if length(mu_movingavg{w}) > 1
            mu_movingavg{w} = alpha*mu_movingavg{w} + (1-alpha)*bn_values.mu{w};
            v_movingavg{w} = alpha*v_movingavg{w} + (1-alpha)*bn_values.v{w};
        else
            mu_movingavg{w} = bn_values.mu{w};
            v_movingavg{w} = bn_values.v{w};
        end
    end
end

function [net_params, val_acc] = MBGD(X, Y, hyperparams, valX, valY, valy, y, net_params, lambda, K)
    n_batch = hyperparams(1);
    eta_min = hyperparams(2);
    eta_max = hyperparams(3);
    n_s = hyperparams(4);
    cycles = hyperparams(5);
    max_t = cycles * 2 * n_s;
    n = size(X, 2);
    train_cost_cycle = zeros(cycles*10,1);
    train_loss_cycle = zeros(cycles*10,1);
    val_cost_cycle = zeros(cycles*10,1);
    val_loss_cycle = zeros(cycles*10,1);
    train_acc = zeros(cycles*10,1);
    val_acc = zeros(cycles*10,1);
    etas = zeros(cycles*10,1);
    mu_movingavg = cell(1, length(net_params.W)-1);
    v_movingavg = cell(1, length(net_params.W)-1);
    plot_t = 0;

    for t = 1:max_t

        batch_index = mod(t, n/n_batch);
        if batch_index ~= 0
            start_index = (batch_index-1)*n_batch + 1;
            end_index = batch_index*n_batch;
        else
            start_index = ((n/n_batch)-1)*n_batch + 1;
            end_index = n;
        end
        eta = compute_clr(eta_min, eta_max, n_s, t);
        Xbatch = X(:, start_index:end_index);
        Ybatch = Y(:, start_index:end_index);
        
        [P, x, bn_values] = EvaluateClassifier(Xbatch, net_params, K);
        gradients = ComputeGradients(Xbatch, Ybatch, P, net_params, bn_values, x, lambda);
        
        %update params
        for w=1:length(net_params.W)
            net_params.W{w} = net_params.W{w} - eta*gradients.W{w};
            net_params.b{w} = net_params.b{w} - eta*gradients.b{w};
        end
        if net_params.use_bn
            [mu_movingavg, v_movingavg] = ComputeMovingAverage(mu_movingavg, v_movingavg, bn_values);
            for w = 1:length(net_params.gammas)
                net_params.gammas{w} = net_params.gammas{w} - eta*gradients.gammas{w};
                net_params.betas{w} = net_params.betas{w} - eta*gradients.betas{w};
            end
        end
        %10 times per cycle
        if mod(t, max_t/cycles/10) == 0 || t == 1
            plot_t = plot_t + 1;
            disp(plot_t)
            %disp("mu" + num2str(mu_movingavg))
            [train_cost_cycle(plot_t), train_loss_cycle(plot_t)] = ComputeCost(X, Y, net_params, lambda, mu_movingavg, v_movingavg);
            [val_cost_cycle(plot_t), val_loss_cycle(plot_t)] = ComputeCost(valX, valY, net_params, lambda, mu_movingavg, v_movingavg);
             val_acc(plot_t) = ComputeAccuracy(valX, valy, net_params, K, mu_movingavg, v_movingavg);
             train_acc(plot_t) = ComputeAccuracy(X, y, net_params, K, mu_movingavg, v_movingavg);
            etas(plot_t) = eta;
        end
    end
    upd_steps = linspace(0,max_t,cycles*10+1);
    % Plot cost
    figure(1)
    plot(upd_steps, train_cost_cycle);
    title('Cost plot')
    xlabel('update step')
    ylabel('cost')
    hold on
    plot(upd_steps, val_cost_cycle);
    legend('train', 'val');
    hold off

    % Plot Accuracy
    figure(2)
    plot(upd_steps, train_acc);
    title('Accuracy plot')
    xlabel('update step')
    ylabel('acc')
    hold on
    plot(upd_steps, val_acc);
    legend('train', 'val');
    hold off

    % Plot loss
    figure(3)
    plot(upd_steps, train_loss_cycle);
    title('Loss plot')
    xlabel('update step')
    ylabel('loss')
    hold on
    plot(upd_steps, val_loss_cycle);
    legend('train', 'val');
    hold off
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)

Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gammas = cell(numel(NetParams.gammas), 1);
    Grads.betas = cell(numel(NetParams.betas), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        c1 = ComputeCost(X, Y, NetTry, lambda);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        c1 = ComputeCost(X, Y, NetTry, lambda);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        c2 = ComputeCost(X, Y, NetTry, lambda);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gammas)
        Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})

            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);

            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);

            Grads.gammas{j}(i) = (c2-c1) / (2*h);
        end
    end

    for j=1:length(NetParams.betas)
        Grads.betas{j} = zeros(size(NetParams.betas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.betas{j})

            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.betas = betas_try;        
            c1 = ComputeCost(X, Y, NetTry, lambda);

            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            c2 = ComputeCost(X, Y, NetTry, lambda);

            Grads.betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end