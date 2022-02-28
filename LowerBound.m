function LB = LowerBound(Wii, Wi, W0, v, N, D, b0, J, S)
    Trii = (S + pinv(W0) + (b0 * N) * J /(b0 + N)) * Wii;
    Trii = (0.5 * v) * trace(Trii);
    Tri = (S + pinv(W0) + (b0 * N) * J /(b0 + N)) * Wi;
    Tri = (0.5 * v) * trace(Tri);
    LB = -1 * lnB(Wii, v, D) - Trii + lnB(Wi, v, D) + Tri;
end