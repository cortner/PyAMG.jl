
# see readme for module documentation
module PyAMG

export RugeStubenSolver,
       SmoothedAggregationSolver,
       solve, solve!, aspreconditioner, set_cycle!


using PyCall

# @pyimport scipy.sparse as scipy_sparse
# @pyimport pyamg
scipy_sparse = pyimport_conda("scipy.sparse", "scipy")
pyamg = pyimport_conda("pyamg", "pyamg")


"""
`py_csc(A::SparseMatrixCSC) -> PyObject`

Takes a Julia CSC matrix and converts it into `PyObject`, which stores a
`scipy.sparse.csc_matrix`.
"""
function py_csc(A::SparseMatrixCSC)
   # create an empty sparse matrix in Python
   Apy = scipy_sparse[:csc_matrix](size(A))
   # write the values and indices
   Apy[:data] = copy(A.nzval)
   Apy[:indices] = A.rowval - 1
   Apy[:indptr] = A.colptr - 1
   return Apy
end


"""
`py_csr(A::SparseMatrixCSC) -> PyObject`

Takes a Julia CSC matrix and converts it into `PyObject`, which stored a
`scipy.sparse.csr_matrix`. (Note: it first converts it to `csc_matrix`, then
calls `tocsr()`)

TODO: this seems extremely inefficient and should at some point be fixed.
"""
py_csr(A::SparseMatrixCSC) = py_csc(A)[:tocsr]()


"""
`type AMGSolver{T}`

Encapsulates the AMG solver types implemented in PyAMG.

Initialise using
```
using PyAMG
amg_rs = RugeStubenSolver(A)
amg_sa = SmoothedAggregationSolver(A)
```
To solve Ax = b:
```
x = solve(amg, b, tol=1e-6, accel="cg")
```
To apply a single MG cycle:
```
x = amg \ b
```
"""
type AMGSolver{T}
    po::PyObject
    id::T
    cycle::AbstractString
    # todo: maxiter
    # todo: tolerance  >>> default arguments, probably use a kwargs Dict
    A::SparseMatrixCSC
end

type RugeStuben end
type SmoothedAggregation end
typealias RugeStubenSolver AMGSolver{RugeStuben}
typealias SmoothedAggregationSolver AMGSolver{SmoothedAggregation}


"""
change the type of AMG cycle, for use with `call` or `\`.
"""
set_cycle!(amg, cycle) = begin amg.cycle = cycle; nothing end


"""
`RugeStubenSolver(A::SparseMatrixCSC, cycle="V", kwargs...)`:

Create a Ruge Stuben instance of `AMGSolver`; wraps
`pyamg.ruge_stuben_solver`
"""
RugeStubenSolver(A::SparseMatrixCSC, cycle="V", kwargs...) =
   AMGSolver(pyamg[:ruge_stuben_solver](py_csr(A), kwargs...),
            RugeStuben(),
            cycle, A)


"""
`SmoothedAggregationSolver(A::SparseMatrixCSC, kwargs...)`

Wrapper for `pyamg.smoothed_aggregation_solver`. See `?AMGSolver` for usage.
"""
SmoothedAggregationSolver(A::SparseMatrixCSC, cycle="V", kwargs...) =
   AMGSolver(pyamg[:smoothed_aggregation_solver](py_csr(A), kwargs...),
            SmoothedAggregation(),
            cycle)


"""
`solve(A::SparseMatrixCSC, b::Vector; kwargs...)`:

PyAMG's 'blackbox' solver. See `pyamg.solve?` for `kwargs`.
"""
solve(A::SparseMatrixCSC, b::Vector; kwargs...) =
    pyamg[:solve]( py_csr(A), b; kwargs... )



"""
`solve(amg::AMGSolver, b, kwargs...)`

Returns a `Vector` with the result of the AMG solver.

### `kwargs`  (copy-pasted from Python docs)

* `x0` : Initial guess.
* `tol` : Stopping criteria: relative residual r[k]/r[0] tolerance.
* `maxiter` : Stopping criteria: maximum number of allowable iterations.
* `cycle` : {"V","W","F","AMLI"}
    Type of multigrid cycle to perform in each iteration.
* `accel` : Defines acceleration method.  Can be a string such as "cg"
    or "gmres" which is the name of an iterative solver in
    `pyamg.krylov` (preferred) or scipy.sparse.linalg.isolve.
    If accel is not a string, it will be treated like a function
    with the same interface provided by the iterative solvers in SciPy.
        (the function version is not tested in Julia!)
* `callback` : User-defined function called after each iteration.  It is
    called as callback(xk) where xk is the k-th iterate vector.
* `residuals` : List to contain residual norms at each iteration.
"""
solve(amg::AMGSolver, b; kwargs...) = amg.po[:solve](b; kwargs...)


######### Capability to use PyAMG.jl as a preconditioner for
######### nonlinear optimisation, sampling, etc

# TODO: the following 4 methods still need to be tested

import Base.\, Base.*
\(amg::AMGSolver, b::Vector) = solve(amg, b)
*(amg::AMGSolver, x::Vector) = amg.A * x

import Base.A_ldiv_B!, Base.A_mul_B!
Base.A_ldiv_B!(x, amg::AMGSolver, b) = solve!(x, amg, b)
Base.A_mul_B!(b, amg::AMGSolver, x) = A_mul_B!(b, amg.A, x)


######### Capability to use PyAMG.jl as a preconditioner for
######### iterative linear algebra

"""
`type AMGPreconditioner`

returned by `aspreconditioner(amg)`, when `amg` is of type `AMGSolver`.
This stores `PyObject` that acts as a linear operator. This type
should be used when `PyAMG` should typically be used as a preconditioner for
iterative linear algebra.

Overloaded methods that can be used with an `AMGPreconditioner` are
`\`, `*`, `A_ldiv_B!`, `A_mul_B!`
"""
type AMGPreconditioner
  po::PyObject
  A::SparseMatrixCSC
end


"""
`aspreconditioner(amg::AMGSolver; cycle=…)`

returns an `M::AMGPreconditioner` object that is suitable for usage
as a preconditioner for iterative linear algebra.

If `x` is a vector, then `M \ x` denotes application of the
preconditioner (i.e. 1 MG cycle), while `M * x` denotes
multiplication with the original matrix from which `amg` was constructed.
"""
aspreconditioner(amg::AMGSolver; cycle=amg.cycle) =
      AMGPreconditioner(amg.po[:aspreconditioner](cycle=cycle), amg.A)

\(amg::AMGPreconditioner, b::Vector) = amg.po[:matvec](b)
*(amg::AMGPreconditioner, x::Vector) = amg.A * x

Base.A_ldiv_B!(x, amg::AMGPreconditioner, b) = copy!(x, amg.po[:matvec](b))
Base.A_mul_B!(b, amg::AMGPreconditioner, x) = A_mul_B!(b, amg.A, x)



"""
`diagnostics(A)`:

Wrapper for `solver_diagnostics`, which is part of PyAMG-Examples. To use it
clone [PyAMG-Examples](https://github.com/pyamg/pyamg-examples), then make the
call to `diagnostics` from a directory where `solver_diagnostics.py` is located
(e.g., from `pyamg-examples/solver_diagnostics`, but the file could be copied
anywhere.

If succesful, `diagnostics(A)` will try a variety of parameter combinations for
`smoothed_aggregation_solver`, and write two files 'solver_diagnostic.txt' and
'solver_diagnostic.py' which contain information how to generate the best
solver.

This implementation is a really poor hack, and suggestions how to improve it
would be highly appreciated.
"""
function diagnostics(A::SparseMatrixCSC; kwargs...)
    # try to import solver_diagnostics
    try
        unshift!(PyVector(pyimport("sys")["path"]), "");
        @pyimport solver_diagnostics
    catch
        error("""
              I tried to @pyimport solver_diagnostics, but it fails.
              This is probably because `solver_diagnostics.py` is not
              in the current directory. Please see
              `?PyAMG.diagonistics` on how to use this function.
              """)
    end

    # import has worked against expectations; call the diagnostics
    solver_diagnostics.solver_diagnostics(A; kwargs...)
end


end
