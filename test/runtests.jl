
print("check wether on travis...")
travis = ccall((:getenv, "libc"), Ptr{UInt8}, (Ptr{UInt8},), "TRAVIS")
if travis != Ptr{UInt8}(0x0000000000000000)
   println("yes: fixing PyCall installation")
   ENV["PYTHON"] = ""
   Pkg.build("PyCall")
else
   println("... no")
end


using PyAMG
using Base.Test

L1d(N) = spdiagm((-ones(N-1), 2*ones(N), -ones(N-1)), (-1,0,1), N, N) * N^2
function L2d(N)
  L1 = L1d(N)
  A = kron(speye(N), L1) + kron(L1, speye(N))
  b = ones(size(A,1))
  return A, b
end
function L3d(N)
  L1 = L1d(N)
  I = speye(N)
  A = kron(L1, I, I) + kron(I, L1, I) + kron(I, I, L1)
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
println("|x_\\ - x_amg|_∞ = ", norm(x3-x1, Inf))
println("|A x_amg - b|_∞ = ", norm(A*x2 - b, Inf))
@assert norm(x3-x1, Inf) < 1e-10

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
@assert norm(x3-x1, Inf) < 1e-9

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
@assert norm(A * x2 - b, Inf) < 1e-8

println("=================================================")
println("Test 4: AMG as a preconditioner")
println("        100 x 100 Dirichlet problem, TOL = 1e-4")
println("        PyAMG vs CG vs PCG  (using IterativeSolvers)")
A, b = L2d(100)
amg = RugeStubenSolver(A)
M = aspreconditioner(amg)
using IterativeSolvers
TOL = 1e-4
println("Plain CG:")
@time x_cg, _ = IterativeSolvers.cg(A, b, 1; tol=TOL)
@time x_cg, _ = IterativeSolvers.cg(A, b, 1; tol=TOL)
println("PyAMG-preconditionerd CG:  (see `aspreconditioner`)")
@time x_pcg, _ = IterativeSolvers.cg(A, b, M; tol=TOL)
@time x_pcg, _ = IterativeSolvers.cg(A, b, M; tol=TOL)
println("PyAMG solver")
@time x_pyamg = PyAMG.solve(amg, b; tol=TOL*1e-2, accel="cg")
@time x_pyamg = PyAMG.solve(amg, b; tol=TOL*1e-2, accel="cg")
x = A \ b
println("|x_cg-x| = ", norm(x_cg - x), " \n",
         "|x_pcg-x| = ", norm(x_pcg - x), "\n",
         "|x_pyamg-x| = ", norm(x_pyamg - x) )
@assert norm(x_cg - x) < 1e-5
@assert norm(x_pcg - x) < 1e-5
@assert norm(x_pyamg - x) < 1e-5
