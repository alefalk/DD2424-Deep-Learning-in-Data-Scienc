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
trainX = allX(:, 1:49000);
trainY = allY(:, 1:49000);
trainy = ally(1:49000, :);
valX = allX(:, 49001:end);
valY = allY(:, 49001:end);
valy = ally(49001:end, :);

%[valX, valY, valy] = LoadBatch("datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_2.mat");
[testX, testY, testy] = LoadBatch("datasets\cifar-10-matlab\cifar-10-batches-mat\test_batch.mat");

mean_trainX = mean(trainX, 2);
std_trainX = std(trainX, 0, 2);

%normalize data
n_trainX = NormalizeData(trainX, mean_trainX, std_trainX);
n_valX = NormalizeData(valX, mean_trainX, std_trainX);
n_testX = NormalizeData(testX, mean_trainX, std_trainX);

m = 50;
d = size(n_trainX, 1);
K = size(trainY, 1);
[W, b] = initialize_network(d, m, K);

eta_min = 1e-5;
eta_max = 1e-1;
n_s = 800;
cycles = 3;
n_batch = 100;
hyperparams = [n_batch, eta_min, eta_max, n_s, cycles];
l_min = -5;
l_max = -1;

% Coarse-Fine search to find best value of lambda.
% CoarseSearch(l_min, l_max, n_trainX, trainY, trainy, n_valX, valY, valy, hyperparams, W, b, 10);
% l_min = 4.772e-5;
% l_max = 4.792e-5;
% num_steps = 10;
% lambda_values = linspace(l_min, l_max, num_steps);
% val_accs(i) = size(lambda_values, 1);
% for i = 1:size(lambda_values, 1)
%     [~, ~, val_acc] = MBGD(n_trainX, trainY, hyperparams, n_valX, valY, valy, trainy, W, b, lambda_values(i));
%     val_accs(i) = val_acc(end);
% end
% matlab.io.saveVariablesToScript('fine_search.m',{'val_accs','lambdas'})

best_lambda = 4.792e-5;


[optimal_W, optimal_b, val_acc] = MBGD(n_trainX, trainY, hyperparams, n_valX, valY, valy, trainy, W, b, best_lambda);
accuracy_test = ComputeAccuracy(n_testX, testy, optimal_W, optimal_b)

function CoarseSearch(l_min, l_max, n_trainX, trainY, trainy, n_valX, valY, valy, hyperparams, W, b, n_lambdas)
    val_accs = zeros(n_lambdas,1);
    lambdas = zeros(n_lambdas,1);
    for i = 1:n_lambdas
        lambda = RandomSampleLambda(l_min, l_max);
        [~, ~, val_acc] = MBGD(n_trainX, trainY, hyperparams, n_valX, valY, valy, trainy, W, b, lambda);
        val_accs(i) = val_acc(end);
        lambdas(i) = lambda;    
    end
    matlab.io.saveVariablesToScript('coarse_search.m',{'val_accs','lambdas'})
end

function lambda = RandomSampleLambda(l_min, l_max)
    l = l_min + (l_max - l_min) * rand(1, 1);
    lambda = 10^l;
end

function eta_t = compute_clr(eta_min, eta_max, n_s, t)
     cycle = floor(1 + t/(2*n_s));
     x = abs(t/n_s - 2*cycle + 1);
     eta_t = eta_min + (eta_max - eta_min) * (max(0, 1-x));
end

function relative_error = ComputeRelativeError(grad_an, grad_num)
    eps = 0.0001;
    relative_error = norm(grad_an - grad_num) / max([eps, norm(grad_an) + norm(grad_num)]);
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
    Y = transpose(Y);
end

function X = NormalizeData(X, mean_trainX, std_trainX) 
    X = X - repmat(mean_trainX, [1, size(X,2)]);
    X = X ./ repmat(std_trainX, [1, size(X,2)]);
end 

function [W, b] = initialize_network(d, m, K)
    % Initialize the weights and biases for a 2-layer neural network.
     W1 = randn(m, d) * sqrt(1/d); % Gaussian distribution with mean 0 and std 1/sqrt(d)
     W2 = randn(K, m) * sqrt(1/m); % Gaussian distribution with mean 0 and std 1/sqrt(m)
     
     b1 = zeros(m, 1);
     b2 = zeros(K, 1);
     
     % double and cell array
     b1 = double(b1);
     b2 = double(b2);
     W = {W1, W2};
     b = {b1, b2};

end

function [P, h] = EvaluateClassifier(X, W, b)
     n = size(X, 2);
     K = size(W{2}, 1);
     P = zeros(K, n);
     s1 = W{1} * X + b{1};
     h = max(0, s1);
     s = W{2} * h + b{2};
     
     for col=1:n
         P(:, col) = softmax(s(:, col));
     
     end
end

 function [grad_W, grad_b] = ComputeGradients(MB, W, lambda, Y, P, h)
     n = size(MB, 2);
     G = -(Y - P);
     grad_W2 = G*transpose(h)/n + 2*lambda*W{2};
     grad_b2 = G*ones(n, 1)/n;
     
     G = transpose(W{2})*G;
     G = G.*(h > 0);
     grad_W1 = G*transpose(MB)/n + 2*lambda*W{1};
     grad_b1 = G*ones(n, 1)/n;
     
     grad_W = {grad_W1, grad_W2};
     grad_b = {grad_b1, grad_b2};

end

function softmax_result = softmax(s)
    softmax_result = exp(s) ./ sum(exp(s), 1);
end

function [J, loss] = ComputeCost(X, Y, W, b, lambda)
    n = size(X, 2);
    C_E_loss = 0;

    for col = 1:n
        image = X(:, col);
        label = Y(:, col);
        C_E_loss = C_E_loss + CrossEntropyLoss(image, label, W, b);
    end
    C_E_loss = C_E_loss / n;
    
    r = 0;
    for w = 1:length(W)
        for i = 1:size(W{w}, 1)
            for j = 1:size(W{w}, 2)
                r = r + W{w}(i, j)^2;
            end
        end
    end
    r = lambda * r;
    loss = C_E_loss;
    J = C_E_loss + r;
end

function C_E_loss = CrossEntropyLoss(image, label, W, b)
    [P, ~] = EvaluateClassifier(image, W, b);
    C_E_loss = -transpose(label) * log(P);
end

function acc = ComputeAccuracy(X, y, W, b)
    n = size(X,2);
    n_errors = 0;
    for col = 1:n
        img = X(:, col);
        [P, ~] = EvaluateClassifier(img, W, b);
        [~, max_p_index] = max(P);
        if max_p_index ~= y(col)
            n_errors = n_errors + 1;
        end
    end
    acc = (length(y) - n_errors) / length(y);
end

function [optimal_W, optimal_b, val_acc] = MBGD(X, Y, hyperparams, valX, valY, valy, y, W, b, lambda)
    n_batch = hyperparams(1);
    eta_min = hyperparams(2);
    eta_max = hyperparams(3);
    n_s = hyperparams(4);
    cycles = hyperparams(5);
    max_t = cycles * 2 * n_s;
    n = size(X, 2);
    train_cost_cycle = zeros(cycles, 1);
    train_loss_cycle = zeros(cycles,1);
    val_cost_cycle = zeros(cycles, 1);
    val_loss_cycle = zeros(cycles, 1);
    train_acc = zeros(cycles, 1);
    val_acc = zeros(cycles, 1);
    etas = zeros(cycles*10,1);

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
        
        [P, h] = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, W, lambda, Ybatch, P, h);
        
        W{1} = W{1} - eta*grad_W{1};
        W{2} = W{2} - eta*grad_W{2};
        b{1} = b{1} - eta*grad_b{1};
        b{2} = b{2} - eta*grad_b{2};

        %10 times per cycle
        if mod(t, max_t/cycles/10) == 0 || t == 1
            plot_t = plot_t + 1;
            [train_cost_cycle(plot_t), train_loss_cycle(plot_t)] = ComputeCost(X, Y, W, b, lambda);
            [val_cost_cycle(plot_t), val_loss_cycle(plot_t)] = ComputeCost(valX, valY, W, b, lambda);
            val_acc(plot_t) = ComputeAccuracy(valX, valy, W, b);
            train_acc(plot_t) = ComputeAccuracy(X, y, W, b);
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

    optimal_W = W;
    optimal_b = b;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
    
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
    
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
end
