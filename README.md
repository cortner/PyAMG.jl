# PyAMG.jl

[![Build Status](https://travis-ci.org/cortner/PyAMG.jl.svg?branch=master)](https://travis-ci.org/cortner/PyAMG.jl)

Convenience wrapper module for the [PyAMG](http://pyamg.org) library.
Requires a Python installation with scipy and pyamg installed.
These must be installed separately, but it is straightforward.


## Basic Usage

In all examples it is assumed that `A` is a sparse matrix
and `b` a vector and `amg` is an `AMGSolver` instance constructed from `A`.

<!-- **Warning:** `amg \ b` does **not** solve $Ax = b$, but it applies a
single multi-grid cycle, typically when AMG is employed as a preconditioner.
To *solve* $Ax = b$, use `x = solve(amg, b)`. -->

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

The following line then performes a single MG cycle
```
p = M \ b
```

E.g., this is compatible with
```
IterativeSolvers.cg(..., Pl=amg, ...)
```

## List of Types and Methods

### Types

* `AMGSolver{T}` : encapsulates the pyamg solver PyObject
* `RugeStubenSolver` : typealias for `AMGSolver{RugeStuben}`
* `SmoothedAggregationSolver` : typealias for `AMGSolver{SmoothedAggregation}`
* `AMGPreconditioner` : encapsulates the output of `aspreconditioner`
   to use PyAMG as a preconditioner for iterative linear algebra.

<!-- ### Methods
* `solve` : basic solver
* `Base.\` : single MG cycle (use PyAMG as preconditioner)
* `set_cycle!` : set which type of cycle to use (default "V")
* `diagnostics` : determine an optimal configuration for a given matrix
* -->
