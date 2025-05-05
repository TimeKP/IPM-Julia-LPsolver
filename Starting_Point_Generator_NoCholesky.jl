using MatrixDepot
using SparseArrays
using JuMP
using LinearAlgebra
using IterativeSolvers

function starting_point(A, b, c)
    # Compute least-squares solution to Ax = b
    # Tilde x, lambda, s
    AAT = A*A'
    x = lsqr(A, b) # Least-squares residual minimizer

    # Compute lambda0 using pseudo-inverse of AAT
    U, S, V = svd(Matrix(AAT))
    inverse_S = diagm(1.0 ./ S)
    pseudo_inverse_AAT = V * inverse_S * U'
    lam = A * c
    lambda0 = pseudo_inverse_AAT * lam

    # Compute s from lambda0
    s = A'*lam
    s = c-s

     # Ensure positivity of x and s
    dx = max(-1.5*minimum(x),0.0)
    ds = max(-1.5*minimum(s),0.0)

    x = x.+dx
    s = s.+ds
    
    # Apply centrality shift
    dx = 0.5*dot(x, s)/sum(s)
    ds = 0.5*dot(x, s)/sum(x)

    x0 = x.+dx
    s0 = s.+ds

    return x0, lambda0, s0
end