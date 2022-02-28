clc;
clear;
data = load('wine.mat');

data_set = data.data;

total_data_number = size(data_set, 1);

train_number = floor(0.8 * total_data_number);

epsilon = 1e-10;

idx = randperm(total_data_number);

X = data_set(idx(1:train_number), 2:size(data_set, 2))';
Y = data_set(idx(1:train_number), 1);

X_test = data_set(idx(train_number+1:end), 2:size(data_set, 2))';
Y_test = data_set(idx(train_number+1:end), 1);

N = train_number;
N_test = total_data_number - N;
D = size(X, 1);


mi = zeros(D, 1);
ma = zeros(D, 1);

for t = 1:D
    mi(t) = min(X(t, :));
    ma(t) = max(X(t, :));
end

for iter = 1:N
    X(:, iter) = (X(:, iter) - ma) ./ (ma - mi);
end

Y_pred = zeros(N, 1);

K = numel(unique(Y));

m0 = zeros(D,1);
m = zeros(D, K);
b0 = 1;

W0 = eye(D, D);
W = zeros(K, D, D);

for c = 1:K
    W(c, :, :) = W0;
end

v0 = D;
v = v0 * ones(K, 1);

H = zeros(K, D, D);

for ii = 1:K
    for jj = 1:D
        H(ii, jj, jj) = 1;
    end
end

xhat = zeros(D, N);

S = zeros(N, D, D); 
J = zeros(K, D, D);

E = zeros(K, D, D);

for c = 1:K
    E(c, :, :) = v0 * W0;
end

n = zeros(K, 1);

Bayes_prior = zeros(K, 1);

Hinv = zeros(K, D, D);

for t = 1:N
    
    n(Y(t)) = n(Y(t)) + 1;
    
    for i = 1:K
%         new_W = 0.5 * (reshape(W(i, :, :), D, D) + reshape(W(i, :, :), D, D)');
%         new_W = reshape(W(i, :, :), D, D);
%         E(i, :, :) = v(i) * new_W;
%         Hinv_reshaped = reshape(E(i, :, :), D, D);
        H_reshaped = reshape(H(i, :, :), D, D);
        Bayes_prior(i) = (n(Y(t)) / t) * mvnpdf(X(:,t), m(:, i), pinv(H_reshaped) + eye(D) * 10);
    end
    
    [~, Y_pred(t)] = max(Bayes_prior);
    if t == 1
        xhat(:, t) = X(:,t);
        S(t, :, :) = (X(:,t) - xhat(:, t)) * (X(:,t) - xhat(:, t))';
    else
        xhat(:, t) = (t-1) * xhat(:, t-1) / t +  X(:,t) / t;
        S(t, :, :) = S(t-1) + t * (X(:,t) - xhat(:, t)) * (X(:,t) - xhat(:, t))' / (t - 1);
    end
    l_t = (Y(t) ~= Y_pred(t));
    
    if l_t > 0
        J(Y(t), :, :) = (xhat(:, t) - m(:, Y(t))) * (xhat(:, t) - m(:, Y(t)))';
        E_reshaped = reshape(E(Y(t), :, :), D, D);
        J_reshaped = reshape(J(Y(t), :, :), D, D);
        W_reshaped = reshape(W(Y(t), :, :), D, D);
        [m(:, Y(t)), H(Y(t), :, :), v(Y(t)), W(Y(t), :, :)] = VIG(t, xhat, X, m(:, Y(t)), b0, v(Y(t)), W_reshaped, E_reshaped, S, J_reshaped, epsilon);
        E(Y(t), :, :) = v(Y(t)) * reshape(W(Y(t), :, :), D, D);
    end
end

Bayes_test = zeros(K,1);
y_pred_test = zeros(N, 1);
for i = 1: N
    for j = 1:K
        H_reshaped = reshape(H(j, :, :), D, D);
        Bayes_test(j) = mvnpdf(X(:,i), m(:, j), pinv(H_reshaped)+ eye(D) * 10);
    end
    [~, y_pred_test(i)] = max(Bayes_test);
end

disp(sum(y_pred_test == Y) / N);



% Bayes_test = zeros(K,1);
% y_pred_test = zeros(N_test, 1);
% 
% for i = 1: N_test
%     for j = 1:K
%         Hinv_reshaped = reshape(E(j, :, :), D, D);
%         Bayes_test(j) = mvnpdf(X_test(:,i), m0(:, end), pinv(Hinv_reshaped));
%     end
%     [~, y_pred_test(i)] = max(Bayes_test);
% end
% 
% disp(sum(y_pred_test == Y_test) / N_test);
