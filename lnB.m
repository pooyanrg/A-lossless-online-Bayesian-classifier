function b = lnB(W, v, D)
    b = -1 * v * 0.5 * log(numel(unique(W))) - v * D * 0.5 * log(2);
    b = b - D * (D - 1) * 0.25 * log(pi);
    for i = 1:D
        b = b - log(gamma((v + 1 - i) * 0.5));
    end
end