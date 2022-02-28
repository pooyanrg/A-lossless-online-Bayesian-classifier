function [m, H, v, W] = VIG(t, xhat, X, m0, b0, v0, W0, E, S, J, epsilon)
epoch = 10000;
N = t;
D = size(X, 1);
m  = zeros(D, 1);
H  = zeros(D, D);
W  = zeros(D, D);
Wlast = W0;

for i = 1:epoch
    m = (b0 * m0 + N * xhat(:, N)) / (b0 + N);
    H = (b0 + N) * E;
    v = (v0 + N + 1);
    W = pinv(pinv(W0) + (b0 + N) * pinv(H) + reshape(S(N, :, :), D, D) + (b0 * N * J) / (b0 + N));
    LB = LowerBound(W, Wlast, W0, v, N, D, b0, J, reshape(S(N, :, :), D, D));
    Wlast = W;
    if i > 1 && LB < epsilon
        break
    end
end