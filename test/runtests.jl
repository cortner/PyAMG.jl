
using PyAMG
using Test

L1d(N) = spdiagm( -1 => -ones(N-1), 0 => 2*ones(N), 1 => -ones(N-1) ) * N^2
function L2d(N)
  L1 = L1d(N)
  ID = sparse(1.0I, N, N)
  A = kron(ID, L1) + kron(L1, ID)
  b = ones(size(A,1))
  return A, b
end
function L3d(N)
  L1 = L1d(N)
  ID = sparse(1.0I, N, N)
  A = kron(L1, ID, ID) + kron(ID, L1, ID) + kron(ID, ID, L1)
  b = ones(size(A,1))
  return A, b
end

println("=================================================")
println("Test 1: Julia `\\` vs PyAMG Blackbox `solve` (2D) Laplacian")
A, b = L2d(100)
print(" \\ : ")
@time x1 = A \ b
print("PyAMG-Warmup: ")
@time x2 = PyAMG.solve(A, b; tol=1e-12, verb=false);
print("PyAMG: ")
@time x3 = PyAMG.solve(A, b; tol=1e-12, verb=false)
println("|x_\\ - x_amg|_∞ = ", norm(x3 - x1, Inf))
println("|A x_amg - b|_∞ = ", norm(A * x2 - b, Inf))
@test norm(x3 - x1, Inf) < 1e-10

println("=================================================")
println("Test 2: Julia `\\` vs PyAMG Blackbox `solve` (3D)")
A, b = L3d(30)
print(" \\ : ")
@time x1 = A \ b
print("PyAMG-Warmup: ")
@time x2 = PyAMG.solve(A, b; tol=1e-12, verb=false)
print("PyAMG: ")
@time x3 = PyAMG.solve(A, b; tol=1e-12, verb=false)
println("|x_\\ - x_amg|_∞ = ", norm(x3-x1, Inf))
println("|A x_amg - b|_∞ = ", norm(A*x2 - b, Inf))
@test norm(x3-x1, Inf) < 1e-9

println("=================================================")
println("Test 3: RugeStubenSolver (3D)")
println("        50 x 50 x 50 grid = 125k dofs ")
A, b = L3d(50)
print("Create solver: ")
@time amg = RugeStubenSolver(A)
print("First solve: ")
@time x1 = solve(amg, b)
print("Second solve: (tol 1e-9)")
@time x2 = solve(amg, b; tol=1e-9)
print("Third solve: (tol 1e-6)")
@time x3 = solve(amg, b; tol=1e-6)
println("|A x_amg - b|_∞ = ", norm(A * x3 - b, Inf))
@test norm(A * x2 - b, Inf) < 1e-8

println("=================================================")
println("Test 4: \\, ldiv, and kwargs")
A, b = L2d(50)
amg = RugeStubenSolver(A)
x_solve = solve(amg, b, tol=1e-6, cycle="V", accel="cg")
set_kwargs!(amg, tol=1e-6, cycle="V", accel="cg")
x_bs = amg \ b
x_ldiv = A_ldiv_B!(similar(b), amg, b)
@test x_solve == x_bs
@test x_solve == x_ldiv

# TODO: IterativeSolvers.jl is not up to date yet
# println("=================================================")
# println("Test 5: AMG as a preconditioner")
# println("        100 x 100 Dirichlet problem, TOL = 1e-4")
# println("        PyAMG vs CG vs PCG  (using IterativeSolvers)")
# A, b = L2d(100)
# amg = RugeStubenSolver(A)
# M = aspreconditioner(amg)
# using IterativeSolvers
# TOL = 1e-4
# println("Plain CG:")
# @time x_cg = IterativeSolvers.cg(A, b; tol=TOL)
# @time x_cg = IterativeSolvers.cg(A, b; tol=TOL)
# println("PyAMG-preconditionerd CG:  (see `aspreconditioner`)")
# @time x_pcg = IterativeSolvers.cg(A, b; Pl=M, tol=TOL)
# @time x_pcg = IterativeSolvers.cg(A, b; Pl=M, tol=TOL)
# println("PyAMG solver")
# @time x_pyamg = PyAMG.solve(amg, b; tol=TOL*1e-2, accel="cg")
# @time x_pyamg = PyAMG.solve(amg, b; tol=TOL*1e-2, accel="cg")
# x = A \ b
# println("|x_cg-x| = ", norm(x_cg - x), " \n",
#          "|x_pcg-x| = ", norm(x_pcg - x), "\n",
#          "|x_pyamg-x| = ", norm(x_pyamg - x) )
# @test norm(x_cg - x) < 1e-5
# @test norm(x_pcg - x) < 1e-5
# @test norm(x_pyamg - x) < 1e-5
