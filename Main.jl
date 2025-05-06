using MatrixDepot
using SparseArrays
using LinearAlgebra
using JuMP
using Dates
using Printf
#using GLPK
#using Clp
#using MathProgBase
#using DataStructures

include("Convert_to_Standard.jl")
include("Starting_Point_Generator_NoCholesky.jl")
include("Starting_Point_Generator_WithCholesky.jl")
include("MehrotraPredictor_Corrector.jl")
include("Newton.jl")
include("CLP_Solver.jl")
include("Backtracking_Line_Search.jl")

mutable struct IplpSolution
    x::Vector{Float64} # the solution vector 
    flag::Bool         # a true/false flag indicating convergence or not
    cs::Vector{Float64} # the objective vector in standard form
    As::SparseMatrixCSC{Float64} # the constraint matrix in standard form
    bs::Vector{Float64} # the right hand side (b) in standard form
    xs::Vector{Float64} # the solution in standard form
    lam::Vector{Float64} # the solution lambda in standard form
    s::Vector{Float64} # the solution s in standard form
end

mutable struct IplpProblem
    c::Vector{Float64}
    A::SparseMatrixCSC{Float64} 
    b::Vector{Float64}
    lo::Vector{Float64}
    hi::Vector{Float64}
end

function convert_matrixdepot(P::MatrixDepot.MatrixDescriptor)
    # key_base = sort(collect(keys(mmmeta)))[1]
    return IplpProblem(vec(P.c), P.A, vec(P.b), vec(P.lo), vec(P.hi))
end

function iplp(problem; max_iter=150, tol = 1e-8, use_mehrotra=true)

    A = problem.A
    b = problem.b
    c = problem.c
    hi = problem.hi
    lo = problem.lo

    m, n = size(A)
    flag = false
    timestamp_initial = now() # Record the initial time
    timestamp_later = nothing # Placeholder for final time

    # Backtracking line search parameters
    alpha = 0.01 # approaching variable
    beta = 0.55 # to reduce alpha

    iter_count = 0
    
    # standard-form conversion: minimize c^Tx s.t. Ax = b, x >= 0
    As, bs, cs = convert_to_standard(A, b, c, hi, lo)

    # Compute a feasible starting point (x > 0, s > 0)
    xs, lam, s = starting_point(As, bs, cs)
    # Alternative: use Cholesky-based initialization for better conditioning
    # xs, lam, s = starting_point_with_cholesky(As, bs, cs; reg_e=1e-7, taumax=1e8)
    
    # Main primal-dual interior-point loop
    for iter = 1:max_iter
        # Increment the iteration count
        iter_count += 1

        # Compute the residuals: primal (r_p), dual (r_d), complementarity (r_g)
        r_p = As * xs .- bs #primal
        r_d = As' * lam .+ s .- cs #dual
        r_g = xs .* s #complementary gap

        # Check the stopping criterion
        du_tol = norm(r_g)/n
        re_tol = norm([r_d; r_p; r_g])/norm([bs; cs])

        if (du_tol < tol && re_tol < tol)
            timestamp_later = now() # Record time upon convergence
            flag = true
            break
        end

        # Compute the search direction via Newton step or Mehrotra's predictor-corrector
        dx, dlambda, ds = get_search_direction(As, xs, s, r_p, r_d, r_g; use_mehrotra=use_mehrotra, reg_e=1e-7)

        # Determine step length using backtracking line search
        t = backtracking_line_search(As, bs, cs, xs, lam, s, dx, dlambda, ds, alpha, beta)

        # Update primal-dual variables
        xs .+= t .* dx
        lam .+= t .* dlambda
        s .+= t .* ds
    end

    # If the method terminated without convergence, still measure total time
    if timestamp_later === nothing
        timestamp_later = now()
    end

    elapsed_time = timestamp_later - timestamp_initial

    return IplpSolution(vec(xs[1:n] .+ lo), flag, vec(cs), sparse(As), vec(bs), vec(xs), vec(lam), vec(s)), iter_count, elapsed_time
end


# -------------------------------------------------------------
# To test an individual LPnetlib problem, uncomment this block:
# -------------------------------------------------------------
# 'lp_afiro','lp_brandy','lp_fit1d','lp_adlittle','lp_agg','lp_ganges','lp_stocfor1', 'lp_25fv47', 'lpi_chemcom'
# raw_problem =  mdopen("LPnetlib/lp_afiro")
# problem = convert_matrixdepot(raw_problem)
# println("Problem = lp_afiro")

# Solve the linear programming problem
# solution, iter_count = iplp(problem)

# println("Optimal value = ", solution.cs'*solution.xs)
# println("Total iterations = ", iter_count)


# Problem list from LPnetlib collection
problems = [
    "lp_afiro", "lp_brandy", "lp_fit1d", "lp_adlittle", "lp_agg",
    "lp_ganges", "lp_stocfor1", "lp_25fv47", "lpi_chemcom",
    #"lp_sc50a", "lp_sc50b", "lp_degen2", "lp_ken_13", "lp_ken_07",
    #"lp_fit2d", "lp_truss"
]

# Print header for comparison table
println("-------------------------------------------------------------------------------------------")
println("| Problem      | Opt(IPM)    | Opt(Clp)    | Time(IPM) | Time(Clp) | Iter(IPM) | Success? |")
println("-------------------------------------------------------------------------------------------")

for pname in problems
    try
        # Load and convert problem from MatrixDepot
        raw_problem = mdopen("LPnetlib/$pname")
        problem = convert_matrixdepot(raw_problem)

        # Solve using our custom Interior-Point Method (IPM) solver
        sol, iter_count, ipm_time = iplp(problem)
        ipm_obj = sol.cs' * sol.xs

        # Solve using reference CLP solver for comparison
        clp_obj, clp_time = solve_with_clp(problem.A, problem.b, problem.c)

        # Print results in table
        println(
            @sprintf("| %-12s | %11.4e | %11.4e | %8.2fs | %8.2fs | %9d | %-8s |",
                pname, ipm_obj, clp_obj,
                convert(Int, Millisecond(ipm_time).value) / 1000,
                convert(Int, Millisecond(clp_time).value) / 1000,
                iter_count,
                sol.flag ? "Yes" : "No")
        )

    catch e
        # Catch and report failure during solution process
        println(@sprintf("| %-12s | ERROR during solving: %s", pname, e))
    end
end

# Print footer
println("------------------------------------------------------------------------------------------")
