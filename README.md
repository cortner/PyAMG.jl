
# PyAMG.jl

[![Build Status](https://travis-ci.org/cortner/PyAMG.jl.svg?branch=master)](https://travis-ci.org/cortner/PyAMG.jl)

Convenience wrapper module for the [PyAMG](https://pyamg.readthedocs.io/en/latest/) library.
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
using SparseArrays, LinearAlgebra
N = 100
L1 = spdiagm( -1 =>  -ones(N-1),
               0 => 2*ones(N),
               1 =>  -ones(N-1) ) * N^2
Id = sparse(1.0*I, N, N)
A = kron(Id, L1) + kron(L1, Id)
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
TOL = 1e-3
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

### Types

* `AMGSolver{T}` : encapsulates the PyAMG solver as a `PyObject`; parameterised
  by a solver-type tag (`RugeStuben` or `SmoothedAggregation`).
* `RugeStubenSolver` : alias for `AMGSolver{RugeStuben}`; wraps
  `pyamg.ruge_stuben_solver`.
* `SmoothedAggregationSolver` : alias for `AMGSolver{SmoothedAggregation}`; wraps
  `pyamg.smoothed_aggregation_solver`.
* `AMGPreconditioner` : returned by `aspreconditioner(amg)`; stores the PyAMG
  linear-operator object for use as a preconditioner in iterative solvers.

### Methods

* `solve(A, b; kwargs...)` : blackbox solver — constructs an AMG hierarchy and
  solves `Ax = b` in one call; wraps `pyamg.solve`.
* `solve(amg, b; kwargs...)` : solve `Ax = b` using an existing `AMGSolver`;
  keyword arguments are merged with any defaults stored in `amg`.
* `set_kwargs!(amg; kwargs...)` : store default keyword arguments in an
  `AMGSolver` instance.  These are used automatically by `\` and `solve`.
* `aspreconditioner(amg; cycle="V")` : return an `AMGPreconditioner`; the
  `cycle` keyword selects the multigrid cycle type (`"V"`, `"W"`, `"F"`, or
  `"AMLI"`).
* `\(amg, b)` : solve `Ax = b` for an `AMGSolver`, or apply one MG cycle for
  an `AMGPreconditioner`.
* `*(amg, x)` : multiply the original matrix `A` by vector `x` (available for
  both `AMGSolver` and `AMGPreconditioner`).
* `ldiv!(x, amg, b)` : in-place version of `\`; works for both solver types.
* `mul!(b, amg, x)` : in-place matrix–vector product with the original `A`.
* `diagnostics(A; kwargs...)` : run `solver_diagnostics` from the
  [pyamg-examples](https://github.com/pyamg/pyamg-examples) repository to find
  an optimal `SmoothedAggregationSolver` configuration; writes
  `solver_diagnostic.txt` and `solver_diagnostic.py` to the current directory.
* `py_csc(A)` : convert a Julia `SparseMatrixCSC` to a `scipy.sparse.csc_matrix`.
* `py_csr(A)` : convert a Julia `SparseMatrixCSC` to a `scipy.sparse.csr_matrix`.
