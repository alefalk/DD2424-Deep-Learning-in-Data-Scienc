clear all
clf
clc

% read in and preprocess data
[trainX, trainY, trainy] = LoadBatch("datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_1.mat");
[valX, valY, valy] = LoadBatch("datasets\cifar-10-matlab\cifar-10-batches-mat\data_batch_2.mat");
[testX, testY, testy] = LoadBatch("datasets\cifar-10-matlab\cifar-10-batches-mat\test_batch.mat");

mean_trainX = mean(trainX, 2);
std_trainX = std(trainX, 0, 2);

%normalize data
n_trainX = NormalizeData(trainX, mean_trainX, std_trainX);
n_valX = NormalizeData(valX, mean_trainX, std_trainX);
n_testX = NormalizeData(testX, mean_trainX, std_trainX);

%Initialize parameters W and b
W = 0.01 * randn(size(trainY, 1), size(trainX, 1));
b = 0.01 * randn(size(trainY, 1), 1);

n_batch = 100;
eta = 0.001;
n_epochs = 40;
lambda = 0.1;

% Testing gradients analytically vs numerically (centered difference
% method)
% h = 0.0001;
% P_testing = EvaluateClassifier(n_trainX(1:20, 1:20), W(:,1:20), b);
% [grad_W, grad_b] = ComputeGradients(n_trainX(1:20, 1:20), trainY(:, 1:20), P_testing, W(:,1:20), lambda);
% [grad_b_num, grad_W_num] = ComputeGradsNumSlow(n_trainX(1:20, 1:20), trainY(:, 1:20), W(:,1:20), b, lambda, h);
% 
% rel_error_W = ComputeRelativeError(grad_W, grad_W_num);
% rel_error_b = ComputeRelativeError(grad_b, grad_b_num);

hyperparams = [n_batch, eta, n_epochs];
[optimal_W, optimal_b] = MBGD(n_trainX, trainY, hyperparams, n_valX, valY, W, b, lambda);
accuracy = ComputeAccuracy(n_testX, testy, optimal_W, optimal_b)


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

function relative_error = ComputeRelativeError(grad_an, grad_num)
eps = 0.0001;
relative_error = norm(grad_an - grad_num) / max([eps, norm(grad_an) + norm(grad_num)]);
end

function X = NormalizeData(X, mean_trainX, std_trainX)
    X = X - repmat(mean_trainX, [1, size(X,2)]);
    X = X ./ repmat(std_trainX, [1, size(X,2)]);
end 

function P = EvaluateClassifier(X, W, b)
    P = softmax(W*X + b);
end

function softmax_result = softmax(s)
    softmax_result = exp(s) ./ sum(exp(s), 1);
end

function [J, loss] = ComputeCost(X, Y, W, b, lambda)
    n = size(X, 2);
    K = size(Y, 1);
    d = size(X, 1);

    C_E_loss = 0;

    for col = 1:n
        image = X(:, col);
        label = Y(:, col);
        C_E_loss = C_E_loss + CrossEntropyLoss(image, label, W, b);
    end
    C_E_loss = C_E_loss / n;
    
    r = 0;
    for i = 1:K
        for j = 1:d
            r = r + W(i, j)^2;
        end
    end
    r = lambda * r;
    J = C_E_loss + r;
    loss = C_E_loss;
end

function C_E_loss = CrossEntropyLoss(image, label, W, b)
    P = softmax(W*image + b);
    C_E_loss = -transpose(label) * log(P);
end

function acc = ComputeAccuracy(X, y, W, b)
    n = size(X,2);
    n_errors = 0;
    for col = 1:n
        img = X(:, col);
        P = softmax(W*img + b);
        [~, max_p_index] = max(P);
        if max_p_index ~= y(col)
            n_errors = n_errors + 1;
        end
    end
    acc = (length(y) - n_errors) / length(y);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    n = size(X, 2);
    G = -(Y - P);
    grad_W = G*transpose(X)./n + 2*lambda*W;
    grad_b = G*ones(n, 1)/n;
end

function [optimal_W, optimal_b] = MBGD(X, Y, hyperparams, valX, valY, W, b, lambda)
    n_batch = hyperparams(1);
    eta = hyperparams(2);
    n_epochs = hyperparams(3);
    n = size(X, 2);
    train_cost_epoch = zeros(n_epochs, 1);
    val_cost_epoch = zeros(n_epochs, 1);
    train_loss_epoch = zeros(n_epochs, 1);
    val_loss_epoch = zeros(n_epochs, 1);


    for i = 1:n_epochs
        for j = 1:n/n_batch
            col_start_index = (j-1)*n_batch + 1;
            col_end_index = j*n_batch;
            Xbatch = X(:, col_start_index:col_end_index);
            Ybatch = Y(:, col_start_index:col_end_index);
            
            P = EvaluateClassifier(Xbatch, W, b);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
            
            W = W - eta*grad_W;
            b = b - eta*grad_b;
        end
        [train_cost_epoch(i), train_loss_epoch(i)] = ComputeCost(X, Y, W, b, lambda);
        [val_cost_epoch(i), val_loss_epoch(i)] = ComputeCost(valX, valY, W, b, lambda);
        
    end
    %cost graph
    figure(1)
    plot(train_cost_epoch);
    title('Cost graph')
    xlabel('Epochs')
    ylabel('Cost')
    hold on
    plot(val_cost_epoch);
    legend('train', 'val');
    hold off
    %loss graph
    figure(2)
    plot(train_loss_epoch);
    title('Loss graph')
    xlabel('Epochs')
    ylabel('Loss')
    hold on
    plot(val_loss_epoch);
    legend('train', 'val');
    hold off

    % % for displaying learnt weight matrices
    % for i=1:10
    %     im = reshape(W(i, :), 32, 32, 3);
    %     s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    %     s_im{i} = permute(s_im{i}, [2, 1, 3]);
    %     subplot(1, 10, i);
    %     imshow(s_im{i})
    % end


    optimal_W = W;
    optimal_b = b;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    [c1, ~] = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    [c2, ~] = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    [c1, ~] = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    [c2, ~] = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
end
end