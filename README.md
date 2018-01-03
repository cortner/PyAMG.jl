
# PyAMG.jl

**NOTE:** the `master` branch is for Julia v0.7 and v1.0 only, for
Julia v0.5 and v0.6 see the `jv0.6` branch.

[![Build Status](https://travis-ci.org/cortner/PyAMG.jl.svg?branch=master)](https://travis-ci.org/cortner/PyAMG.jl)

Convenience wrapper module for the [PyAMG](http://pyamg.org) library.
Requires a Python installation with `scipy` and `pyamg`.
If an Anconda distribution is used (including the `Conda.jl` package manager)
then `pyamg` should be automatically installed on first use. Otherwise, follow
the [straightforward instructions](https://github.com/pyamg/pyamg).

<!-- *Note on failing tests:* tests on travis-ci fail, but this is due to
failure of autmatically installing the required packages. All tests pass
under both v0.4 and v0.5 on my own machine. -->

## Basic Usage

In all examples it is assumed that `A` is a sparse matrix
and `b` a vector and `amg` is an `AMGSolver` instance constructed from `A`.
The classical example would be the Dirichlet problem on a square,
```julia
N = 100
L1 = spdiagm((-ones(N-1), 2*ones(N), -ones(N-1)), (-1,0,1), N, N) * N^2
A = kron(speye(N), L1) + kron(L1, speye(N))
b = ones(size(A,1))
```

### Blackbox solver
```julia
using PyAMG
x = PyAMG.solve(A, b)
```

### Multiple solves

To initialise, call
```julia
using PyAMG
amg = RugeStubenSolver(A)
```

Then the system Ax=b can be solved using
```julia
x = amg \ b
x = solve(amg, b; tol=1e-6)
```

or, one can specify a different 'outer solver'
```julia
x = solve(amg, b; tol=1e-6, accel="cg")
```

see `?solve` for more options. In particular, note the that default keyword
arguments can be set via `set_kwargs!` or in the construction of the AMG
solver, which will then be used by both `\` and `solve`. E.g.,
```julia
amg = RugeStubenSolver(A, tol=1e-6, accel="cg")
x = amg \ b
```

### As Preconditioner

After initialising, we can construct a preconditioner via
```julia
M = aspreconditioner(amg)
```

The line `M \ b` then performes a single MG cycle.
This is e.g. compatible with the `IterativeSolvers` package:
```julia
using PyAMG, IterativeSolvers
M = aspreconditioner(RugeStubenSolver(A))
IterativeSolvers.cg(A, b, M; tol=TOL)
```

### Solver history

To extract the solver history as a vector of residuals, use
```
amg = RugeStubenSolver(A)
r = Float64[]
x = PyAMG.solve(amg, b, residuals=r)
@show r
```
Since version `3.2.1.dev0+2227b77` the residuals can also be returned for
the blackbox solver variant.

(NOTE: although `pyamg` needs residuals to be a *list*, `PyAMG.jl` will detect
   if `residuals` is a `numpy` vector and replace it with a list, then
   convert back to a types Julia vector.)

## List of Types and Methods

(this section is incomplete)

### Types

* `AMGSolver{T}` : encapsulates the pyamg solver PyObject
* `RugeStubenSolver` : typealias for `AMGSolver{RugeStuben}`
* `SmoothedAggregationSolver` : typealias for `AMGSolver{SmoothedAggregation}`
* `AMGPreconditioner` : encapsulates the output of `aspreconditioner`
   to use PyAMG as a preconditioner for iterative linear algebra.

<!-- ### Methods  TODO: write this documentation.
* `solve` : basic solver
* `Base.\` : single MG cycle (use PyAMG as preconditioner)
* `set_cycle!` : set which type of cycle to use (default "V")
* `diagnostics` : determine an optimal configuration for a given matrix
* -->
