clear all
clc

%read in data
book_fname = "data/goblet_book.txt";
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data);

% constants
K = length(book_chars);

char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');
for char_idx = 1:K
    char_to_ind(book_chars(char_idx)) = char_idx;
    ind_to_char(char_idx) = book_chars(char_idx);
end

% Hyperparameters
h = 0.0001;
m = 100;
rng(500)
seq_length = 25;
sig = 0.01;
h0 = zeros(m,1);
epochs = 2;
n_updates = floor(epochs*length(book_data)/seq_length);

%initialize RNN parameters
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

X_chars = book_data(1:end);
Y_chars = book_data(1:end);
onehot_X = onehot_encode(X_chars, K, char_to_ind);
onehot_Y = onehot_encode(Y_chars, K, char_to_ind);

% Train RNN
disp('Training...');
trainedRNN = trainNetwork(RNN, onehot_X, onehot_Y, h0, seq_length, 300000, ind_to_char, book_data);

function Generate(RNN, h0, x_0, seq_length, ind_to_char)
    Y = synthesize_sequence(RNN, h0, x_0, seq_length);
    generated_text = zeros(1, size(Y,2));
    for t = 1:size(Y,2)
        index = find(Y(:, t));
        generated_text(t) = ind_to_char(index);
    end
    generated_text = char(generated_text);
    disp(generated_text);
end

function [RNN, m] = AdaGrad(Grads, RNN, m)
    eta = 0.1;
    epsilon = 1e-8;
    for f = fieldnames(RNN)'
        m.(f{1}) = m.(f{1}) + Grads.(f{1}).^2;
        RNN.(f{1}) = RNN.(f{1}) - eta*(Grads.(f{1}) ./ (m.(f{1}) + epsilon).^(0.5));
    end
end

function m = m_start(RNN)
    for f = fieldnames(RNN)'
        m.(f{1}) = zeros(size(RNN.(f{1})));
    end
end


function trainedRNN = trainNetwork(RNN, onehot_X, onehot_Y, h0, seq_length, n_updates, ind_to_char, book_data)
    e = 1;
    smooth_loss_aggregate = zeros(n_updates, 1);
    m = m_start(RNN);

    for i=1:n_updates
        if e==1
            hprev = h0;
        end

        X_Seq = onehot_X(:, e:e+seq_length-1);
        Y_Seq = onehot_Y(:, e+1:e+seq_length);

        if mod(i, 10000) == 0
            disp("iter: " + i + " smooth loss: " + smooth_loss);
            Generate(RNN, hprev, X_Seq(:, 1), 200, ind_to_char);
        end

        [Grads, hprev, loss] = backProp(RNN, X_Seq, Y_Seq, hprev);

        if i == 1
            smooth_loss = loss;
            disp("iter: " + i + " smooth loss: " + smooth_loss);
            Generate(RNN, h0, X_Seq(:,1), 200, ind_to_char);
        end

        smooth_loss = 0.999*smooth_loss + 0.001 * loss;
        smooth_loss_aggregate(i) = smooth_loss;

        [RNN, m] = AdaGrad(Grads, RNN, m);    

        e = e + seq_length;
        if e>length(book_data)-seq_length-1
            e = 1;
            hprev = h0;
        end
    end

    trainedRNN = RNN;
    upd_steps = linspace(1, n_updates, n_updates);
    plot(upd_steps, smooth_loss_aggregate);
    title('Smooth loss Evolution')
    xlabel('Upd Steps')
    ylabel('Smooth loss')
end

function one_hot_char = onehot_char(next_char_idx, K)
    one_hot_char = zeros(K, 1);
    one_hot_char(next_char_idx, 1) = 1;
end

function one_hots = onehot_encode(chars, K, char_to_ind)
    one_hots = zeros(K, length(chars));
    for char_i = 1:length(chars)
        one_hots(:, char_i) = onehot_char(char_to_ind(chars(char_i)), K);
    end
end

function Y = synthesize_sequence(RNN, h0, x0, n)
    K = size(RNN.c, 1);
    hprev = h0;
    xnext = x0;
    Y = zeros(K, n);
    
    for t = 1:n
        [P_t, hprev] = calculateProbVec(RNN, hprev, xnext);
        next_index = selectRandom(P_t);    
        xnext = onehot_char(next_index, K);
        Y(:, t) = xnext;
    end
end
% eq. (1) - (4)
function [P_t, hnext] = calculateProbVec(RNN, h, x)
    a = RNN.W * h + RNN.U * x + RNN.b;
    hnext = tanh(a);
    o = RNN.V * hnext + RNN.c;
    P_t = softmax(o);
end

function ii = selectRandom(P_t)
    cp = cumsum(P_t);
    a = rand;
    ixs = find(cp-a > 0);
    ii = ixs(1);
end

function softmax_result = softmax(s)
    softmax_result = exp(s) ./ sum(exp(s), 1);
end

function [loss, output_vectors, P] = Forward(RNN, onehot_X, onehot_Y, h0)
    loss = 0;
    n = size(onehot_X, 2);
    output_vectors = zeros(size(h0, 1), n);
    P = zeros(size(onehot_X));
    h = h0;
    for t=1:n
        [P_t, h] = calculateProbVec(RNN, h, onehot_X(:, t));
        loss = loss - log(transpose(onehot_Y(:, t)) * P_t);
        output_vectors(:, t) = h;
        P(:, t) = P_t;
    end
end

%backward pass from Lecture 9
function [Grads, H, loss] = backProp(RNN, onehot_X, onehot_Y, h0)
    Grads.b = zeros(size(RNN.b));
    Grads.c = zeros(size(RNN.c));
    Grads.U = zeros(size(RNN.U));
    Grads.W = zeros(size(RNN.W));
    Grads.V = zeros(size(RNN.V));
    n = size(onehot_X, 2);
    [loss, H, P] = Forward(RNN, onehot_X, onehot_Y, h0);

    for t=n:-1:1
        dL_ot = -transpose(onehot_Y(:, t) - P(:, t));

        Grads.c = Grads.c + transpose(dL_ot);
        Grads.V = Grads.V + transpose(dL_ot) * transpose(H(:, t));

        dL_ht = dL_ot * RNN.V;
        if t~=n
            dL_ht = dL_ht + dL_at * RNN.W;
        end

        Grads.b = Grads.b + diag(1 - H(:, t).^2) * transpose(dL_ht);
        dL_at = dL_ht * diag(1-H(:, t).^2);
        if t~=1
            hprev = H(:, t-1);
        else 
            hprev = h0;
        end
        Grads.W = Grads.W + transpose(dL_at) * transpose(hprev);
        Grads.U = Grads.U + transpose(dL_at) * transpose(onehot_X(:, t));
    end

    %Clipping gradients
    for f = fieldnames(Grads)'
        Grads.(f{1}) = max(min(Grads.(f{1}), 5), -5);
    end

    H = H(:, n);
end

% ---------------------------------------------------------------------- %

function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        [l1, ~, ~] = Forward(RNN_try, X, Y, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        [l2, ~, ~] = Forward(RNN_try, X, Y, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end
