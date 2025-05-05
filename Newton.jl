using LinearAlgebra

function compute_newton_direction(A, x, s, r_p, r_d, r_g; reg_e::Float64 = 1e-7)

    # Dimensions
    m, n = size(A)

    # KKT matrix (Jacobian of the perturbed KKT conditions used in Newton step)
    #    [  0    A    0 ]
    #    [  A'   0    I ]
    #    [  0    S    X ]
    J = [spzeros(m,m) A spzeros(m,n);
         A' spzeros(n,n) Matrix{Float64}(I,n,n);
         spzeros(n,m) spdiagm(0=>s[:,1]) spdiagm(0=>x[:,1])]

    # Compute the steps 
    m = length(r_p)
    n = length(r_d)

    # Solve the Newton system J * Δ = -[r_p; r_d; r_g] to find the search direction
    Fc = Array{Float64}([-r_p; -r_d; -r_g])

    # Choose one of the following methods to solve the linear system J * b = Fc
    b = nothing

    # 1. Try LU decomposition first
    #println("[Newton] try LU")
    try 
        J_f = lu(J)
        b = J_f \ Fc
        #println("[Newton] LU succeeded")
    catch e
        if isa(e, SingularException)
            # 2. On singularity, add regularization to J and retry LU
            #println("[Newton] LU failed (Singular), add regularization ε=$reg_e")
            J += spdiagm(0 => fill(reg_e, size(J, 1)))
            #println("[Newton] retry LU with regularization")
            try
                J_f = lu(J)
                b = J_f \ Fc
                #println("[Newton] Regularized LU succeeded")
            catch
                # 3. If still failing, fallback to QR
                #println("[Newton] Regularized LU also failed, try QR")
                Q, R = qr(Matrix(J))
                try
                    b = R \ (Q' * Fc)
                    #println("[Newton] QR succeeded")
                catch
                    # 4. If still failing, fallback to SVD
                    #    J = U * Σ * Vᵀ  →  pseudoinverse = V * Σ⁻¹ * Uᵀ
                    #println("[Newton] QR failed, try SVD")
                    U, S, V = svd(Matrix(J))
                    inverse_S = diagm(1.0 ./ S)
                    b = V * inverse_S * U' * Fc
                    #println("[Newton] SVD succeeded")
                end
            end
        else
            rethrow(e) # For other exceptions, propagate the error
        end
    end
    
    # Split the components of the direction vector
    dx = b[1+m:m+n]
    dlambda = b[1:m]
    ds = b[1+m+n:m+2*n]

    return dx, dlambda, ds
end
