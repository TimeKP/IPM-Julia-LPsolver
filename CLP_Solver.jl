using JuMP
using GLPK
using MatrixDepot
using SparseArrays
using Dates
using Clp

function solve_with_clp(A, b, c)
    # Record the initial time
    start_time = now()

    # Initialize a JuMP model using the Clp solver
    model = Model(Clp.Optimizer)

    # Suppress solver output for cleaner benchmarking
    set_optimizer_attribute(model, "LogLevel", 0)

    # Define the decision variables: x >= 0
    n = length(c)
    @variable(model, x[1:n] >= 0)

    # Set the objective: minimize c^Tx
    @objective(model, Min, dot(c, x))
    for i in 1:size(A, 1)
        @constraint(model, dot(A[i, :], x) == b[i])
    end
    
    # Solve
    optimize!(model)

    # Compute total elapsed time
    elapsed_time = now() - start_time

    return objective_value(model), elapsed_time
end