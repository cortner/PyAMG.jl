
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
A, b = L3d(100)
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

println("Testing amg \\ b: (PyAMG as a preconditioner)")
println("Cost of amg \\ b: (2 runs)")
@time amg \ b;
@time amg \ b;
