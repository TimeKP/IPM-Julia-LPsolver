using MatrixDepot
using SparseArrays
using JuMP
using LinearAlgebra
using IterativeSolvers

function is_posdef(A)
    try
        F = cholesky(A; throw_indefinite_error=true)
    catch ArgumentError
        return false
    end
    return true
end 

function bisection_to_positive_definite(A; taumax=1e9)
    # check the input 
    tau = 1.0 
    if is_posdef(A)
        return 0.0 
    else
        while is_posdef(A + tau*I) == false 
            tau = 2*tau 
            if tau > taumax 
                throw(ArgumentError("the matrix has a strong negative direction"))
            end 
        end 
        return tau 
    end
end 

function starting_point_with_cholesky(As, bs, cs; reg_e, taumax)
    m, n = size(As)

    x = lsqr(As, bs)

    M = As * As'
    tau = try
        bisection_to_positive_definite(M; taumax=taumax)
    catch e
        @warn "Bisection to SPD failed, fallback to tau=$reg_e"
        reg_e
    end
    M_reg = M + tau * I

    L = cholesky(M_reg)
    lam = L \ (As * cs)

    s = cs - As' * lam

    mu = dot(x, s) / n

    dx = max.(0.5 * mu ./ s, -minimum(x) + 1e-8)
    ds = max.(0.5 * mu ./ x, -minimum(s) + 1e-8)

    x = x.+dx
    s = s.+ds
    
    return x, lam, s
end
