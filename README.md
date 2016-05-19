# PyAMG.jl

[![Build Status](https://travis-ci.org/cortner/PyAMG.jl.svg?branch=master)](https://travis-ci.org/cortner/PyAMG.jl)

Convenience wrapper module for the [PyAMG](http://pyamg.org) library.
Requires a Python installation with scipy and pyamg installed.
These must be installed separately, but it is straightforward.

## Basic Usage

In all examples it is assumed that `A` is a sparse matrix
and `b` a vector.

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
x = solve(amg, b; tol=1e-6)
```
or, one can specify a different 'outer solver'
```
x = solve(amg, b; tol=1e-6, accel="cg")
```

### As Preconditioner

After initialising, the following line performes a single MG cycle
```
p = amg \ b
```
E.g., this is compatble with
```
IterativeSolvers.cg(..., Pl=amg, ...)
```

## List of Types and Methods

### Types

* `AMGSolver{T}` : encapsulates the pyamg solver PyObject
* `RugeStubenSolver` : typealias for `AMGSolver{RugeStuben}`
* `SmoothedAggregationSolver` : typealias for
 `AMGSolver{SmoothedAggregationSolver}`

### Methods

* `solve` : basic solver
* `Base.\` : single MG cycle (use PyAMG as preconditioner)
* `set_cycle!` : set which type of cycle to use (default "V")
* `diagnostics` : determine an optimal configuration for a given matrix
