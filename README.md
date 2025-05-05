# Interior-Point LP Solver in Julia
This project implements a primal-dual interior-point method (IPM) for solving linear programming (LP) problems in Julia. The solver is modular, numerically stable, and supports bounded-variable LPs, which are automatically converted into standard form. It optionally supports Mehrotra's predictor-corrector method for accelerated convergence.

## Features
- Conversion from bounded-variable LPs to standard form
- Newton direction solver with fallback to LU, QR, or SVD
- Optional Mehrotra predictor-corrector step
- Backtracking line search
- Compatible with `MatrixDepot.jl` for benchmarking on LPnetlib problems
- Comparison against Clp solver baseline

## Module Overview
| Module/File                        | Description |
|-----------------------------------|-------------|
| `Convert_to_Standard.jl`          | Converts bounded LPs to standard form |
| `Starting_Point_Generator_NoCholesky.jl` | Computes feasible initial point |
| `Newton.jl`                       | Computes Newton direction with regularization |
| `MehrotraPredictor_Corrector.jl` | Implements Mehrotra's method (optional) |
| `Backtracking_Line_Search.jl`    | Step size selection |
| `CLP_Solver.jl`                  | Clp baseline comparison |
| `Main.jl`                         | Integrates all modules and runs `iplp()` |

## Running a Specific LPnetlib Example
To run a **specific** LPnetlib example (instead of the full benchmark suite), you can modify the `Main.jl` file:

1. Uncomment the following block:

```julia
# -------------------------------------------------------------
# To test an individual LPnetlib problem, uncomment this block:
# -------------------------------------------------------------
# 'lp_afiro','lp_brandy','lp_fit1d','lp_adlittle','lp_agg','lp_ganges','lp_stocfor1', 'lp_25fv47', 'lpi_chemcom'
# problem =  mdopen("LPnetlib/lp_afiro")
# println("Problem = lp_afiro")

# Solve the linear programming problem
# solution, iter_count = iplp(problem)

# println("Optimal value = ", solution.cs'*solution.xs)
# println("Total iterations = ", iter_count)
```

2. Replace `"lp_afiro"` with any other LPnetlib problem name (e.g., `"lp_ganges"`).

3. Run `Main.jl` in your Julia environment.

## Requirements
- Julia 1.6+
- Packages:
  - `MatrixDepot.jl`
  - `JuMP.jl`
  - `SparseArrays`
  - `Clp.jl` (for comparison)

Install with:
```julia
import Pkg
Pkg.add(["MatrixDepot", "JuMP", "Clp"])
```

## Benchmarking
The solver has been tested on LP problems from the LPnetlib collection, including:
- `lp_afiro`, `lp_brandy`, `lp_adlittle`, `lp_agg`, `lp_ganges`, `lp_stocfor1`, `lp_25fv47`, `lpi_chemcom`

Performance is measured in terms of:
- Number of iterations to convergence
- Runtime
- Feasibility and optimality residuals
