# PyAMG.jl

[![Build Status](https://travis-ci.org/cortner/PyAMG.jl.svg?branch=master)](https://travis-ci.org/cortner/PyAMG.jl)

Convenience wrapper module for the [PyAMG](http://pyamg.org) library.
Requires a Python installation with `scipy` and `pyamg`.
If an Anconda distribution is used (including the `Conda.jl` package manager)
then `pyamg` should be automatically installed on first use. Otherwise, follow
the [straightforward instructions](https://github.com/pyamg/pyamg).

*Note on failing tests:* tests on travis-ci fail, but this is due to
failure of autmatically installing the required packages. All tests pass
under both v0.4 and v0.5 on my own machine.

## Basic Usage

In all examples it is assumed that `A` is a sparse matrix
and `b` a vector and `amg` is an `AMGSolver` instance constructed from `A`.
The classical example would be the Dirichlet problem on a square,
```
N = 100
L1 = spdiagm((-ones(N-1), 2*ones(N), -ones(N-1)), (-1,0,1), N, N) * N^2
A = kron(speye(N), L1) + kron(L1, speye(N))
b = ones(size(A,1))
```

### Blackbox solver
```
using PyAMG
x = PyAMG.solve(A, b)
```

### Multiple solves

To initialise, call
```
using PyAMG
amg = RugeStubenSolver(A)
```

Then the system Ax=b can be solved using
```
x = amg \ b
x = solve(amg, b; tol=1e-6)
```

or, one can specify a different 'outer solver'
```
x = solve(amg, b; tol=1e-6, accel="cg")
```

see `?solve` for more options.

### As Preconditioner

After initialising, we can construct a preconditioner via
```
M = aspreconditioner(amg)
```

The line `M \ b` then performes a single MG cycle.
This is e.g. compatible with the `IterativeSolvers` package:
```
using PyAMG, IterativeSolvers
M = aspreconditioner(RugeStubenSolver(A))
IterativeSolvers.cg(A, b, M; tol=TOL)
```

## List of Types and Methods

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
