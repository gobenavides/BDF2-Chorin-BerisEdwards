# Simulation: defect dynamics

from firedrake import *
import numpy as np
import finat
import csv
from mpi4py import MPI

symmetry = True # If we want the space for Q and H to be strongly symmetric.

solver_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

#################################### Mesh, geometric and physical quantities ####################################

nn = 6

dom_length = 2
mesh = SquareMesh(2**nn, 2**nn, dom_length) # [0,2]^2
x, y = SpatialCoordinate(mesh)
n_vec = FacetNormal(mesh)
h_max = mesh.comm.allreduce(mesh.cell_sizes.dat.data.max(), op=MPI.MAX)

d = 2 # Space dimension

### Physical parameters
a = -0.2
b = 1
c = 1
A0 = 500

mu = 1
xi = 0
M = 10
L = 1e-3

### Numerical parameters
T = 250             # End time
dt = 1/500
t = Constant(0.0)   # Current time

################################## Initialize CSV for Energy
csv_filename = "energies.csv"
if mesh.comm.rank == 0:
	with open(csv_filename, mode='w', newline='') as file:
		writer = csv.writer(file, delimiter='\t')
		writer.writerow(["Time", "Continuous_Energy", "Discrete_Energy"])

# Function to export to CSV
def record_energy(time_val, E_c, E_d):
	PETSc.Sys.Print(f"Time {float(time_val):10.4f}, \tContinuous energy: {E_c:10.16f}, \tDiscrete energy: {E_d:10.16f}")
	if mesh.comm.rank == 0:
		with open(csv_filename, mode='a', newline='') as file:
			writer = csv.writer(file, delimiter='\t')
			writer.writerow([float(time_val), E_c, E_d])

#################################### Operators ####################################

def aux_var(A):
	return sqrt(2*( (a/2)*tr(dot(A,A)) -(b/3)*tr( dot(A,dot(A,A)) ) + (c/4)*tr(dot(A,A))**2 + A0 ) )

def sigma(Q,H):
	return dot(Q,H) - dot(H,Q) - xi*(dot(H,Q) + dot(Q,H)) - 2*xi/d*H + 2*xi*inner(Q,H)*Q

def sym_grad(u):
	return 0.5*(grad(u) + grad(u).T)

def skew_grad(u):
	return 0.5*(grad(u) - grad(u).T)

def S(u,Q): # This is little s
	DD = sym_grad(u)
	WW = skew_grad(u)

	return dot(WW,Q) - dot(Q,WW) + xi*(dot(Q,DD) + dot(DD,Q)) \
	+ (2*xi/d)*DD - (2*xi/d**2)*div(u)*Identity(d) - 2*xi*inner(DD,Q)*(Q + (1/d)*Identity(d))

def P(Q):
	V = a*Q - b*(dot(Q,Q) - (1/d)*tr(dot(Q,Q))*Identity(d)) + c*tr(dot(Q,Q))*Q
	return V/aux_var(Q)

def conv_term(uu,vv,ww):
	return (inner(dot(grad(vv),uu),ww) + 0.5*div(uu)*inner(vv,ww))

def extrap(B0,B1):
	return 2*B1 - B0

def contract(RR,vv):
	i,j,k = indices(3)
	return as_tensor( RR[i,j,k]*vv[k], (i,j))

def contract_higher(AA,RR):
	i,j,k = indices(3)
	return as_vector( AA[i,j]*RR[i,j,k], k)

### Problem data
epsilon = Constant(1e-10) # To avoid division by zero

u_bc = as_vector([0,0])
n_bc = as_vector([x-0.5*dom_length, y-0.5*dom_length])
Q_bc_proto = outer(n_bc,n_bc)/(inner(n_bc,n_bc)+epsilon)
Q_bc = Q_bc_proto - (tr(Q_bc_proto)/d)*Identity(d) # Traceless symmetric
r_bc = aux_var(Q_bc)

u_in = as_vector([x-x,0])
n_in = as_vector([x-0.25*dom_length, y-0.25*dom_length])
Q_in_proto = outer(n_in,n_in)/(inner(n_in,n_in)+epsilon)
Q_in = Q_in_proto - (tr(Q_in_proto)/d)*Identity(d) # Traceless symmetric
r_in = aux_var(Q_in)

#################################### Finite Element spaces ####################################

# Taylor-Hood pair
U_h = VectorFunctionSpace(mesh, "CG", 2, dim=2) # Intermediate velocity space # Don't forget about the bcs!
P_h = FunctionSpace(mesh, "CG", 1)              # Pressure space, we should enforce mean-zero!

X_h = FunctionSpace(mesh, "CG", 1)              # Auxiliary variable space
M_h = TensorFunctionSpace(mesh, "CG", 1, symmetry=symmetry) # Space for Q-tensor (should I assume it is symmetric?)
# Y_h = U_h ⊕ ∇P_h. Each function in Y_h can be uniquely decomposed!

### Mixed FE

# Space for the hydrodynamics
FES_hydro = U_h * M_h * M_h * X_h
bc_u = DirichletBC(FES_hydro.sub(0), u_bc, (1,2,3,4))
bc_Q = DirichletBC(FES_hydro.sub(1), Q_bc, (1,2,3,4))
bc_hydro = [bc_u, bc_Q]

v, Z, Y, w = TestFunctions(FES_hydro) # Some test functions. We will use them many times

# Space for the projection steps (with explicit communicator to silence MPI warnings)
FES_proj = U_h * P_h * P_h
nullspace = MixedVectorSpaceBasis(
	FES_proj, [
		FES_proj.sub(0),
		VectorSpaceBasis(constant=True, comm=mesh.comm),
		VectorSpaceBasis(constant=True, comm=mesh.comm)
	]
) # To remove constants in the pressure space
bc_proj = DirichletBC(FES_proj.sub(0), u_bc, (1,2,3,4)) # Dirichlet boundary condition

z, hat_q, q = TestFunctions(FES_proj) # Some test functions. We will use them many times

################################### For visualization ##############################################
outfile = VTKFile("exp_1.pvd")

# Dedicated spaces and persistent functions for ParaView visualization
U_out_space = VectorFunctionSpace(mesh, "CG", 2, dim=2)
u_out = Function(U_out_space, name="Velocity")
p_out = Function(P_h, name="Pressure")
Q_out = Function(M_h, name="Q")
H_out = Function(M_h, name="H")
r_out = Function(X_h, name="r")

S_out_space = FunctionSpace(mesh, "CG", 1)
S_out = Function(S_out_space, name="Order_Parameter")

def export_to_pvd(u_val, p_val, Q_val, H_val, r_val, time_val):
	"""Helper to safely interpolate expressions and write native functions directly to VTK."""

	# Interpolate/assign current values into the permanently named output buffer
	u_out.interpolate(u_val)
	p_out.interpolate(p_val)  # Using .interpolate() safely evaluates any UFL expressions!
	Q_out.assign(Q_val)       # .assign() is faster than .interpolate() for identical FE spaces
	H_out.assign(H_val)
	r_out.assign(r_val)

	# Compute Order Parameter S directly via UFL
	S_out.interpolate(2 * sqrt(Q_val[0, 0]**2 + Q_val[0, 1]**2))

	# Pass native functions (including Q_val) directly to outfile.write
	outfile.write(u_out, p_out, Q_out, H_out, r_out, S_out, time=float(time_val))

#################################### Discrete initial conditions ####################################
def disc_ic_0(): # t = 0
	# L2 Projection of u_in on U_h
	u_tilde_0 = project(u_in, U_h, bcs=DirichletBC(U_h, u_bc, (1,2,3,4)))

	# Projection of u_tilde_0 in the "weakly" divergence functions of V_h.
	yp_proj_p = Function(FES_proj)
	y_proj_trial, p_proj_trial, p_trial = split(yp_proj_p)

	F_proj = (1/dt)*inner(y_proj_trial+grad(p_proj_trial)-u_tilde_0, z+grad(hat_q))*dx(degree=4) + inner(grad(p_trial), z+grad(hat_q))*dx(degree=4) + inner(grad(q), y_proj_trial+grad(p_proj_trial))*dx(degree=4)
	solve(F_proj==0, yp_proj_p, nullspace=nullspace, bcs=bc_proj, solver_parameters=solver_parameters)

	y_proj_0, p_proj_0, p_0 = yp_proj_p.subfunctions # We care about u_h = y_h + ∇p̂ₕ, and the pressure pₕ

	p_0normalized = p_0 - assemble(p_0*dx(degree=4))/assemble(1*dx(mesh))

	# L2 Projection of Q_in in M_h
	Q_0 = Function(M_h).interpolate(Q_in)
	# Q_0 = project(Q_in, M_h, bcs=DirichletBC(M_h, Q_bc, (1,2,3,4))) # Project Q_in to the M_h and enforce boundary conditions. Notice that Q_in ≠ Q_bc on the boundary.

	# Lagrange interpolant of r(Q_in)
	r_0 = Function(X_h)
	r_0.interpolate(r_in)

	return u_tilde_0, y_proj_0, p_proj_0, Q_0, r_0, p_0normalized

u_tilde_0, y_proj_0, p_proj_0, Q_0, r_0, p_0 = disc_ic_0()
u_0 = y_proj_0 + grad(p_proj_0) # Temporary definition for t=0 export

# --- Export initial state (t = 0) ---
H_0 = Function(M_h, name="H")   # Placeholder zero-function since H is not solved at t=0
export_to_pvd(u_0, p_0, Q_0, H_0, r_0, time_val=0.0)

# --- Compute and log t=0 energy ---
E_tot_0 = assemble((0.5*r_0**2 - A0)*dx(degree=4) + 0.5*L*inner(grad(Q_0), grad(Q_0))*dx(degree=4))
E_disc_0 = assemble(2*inner(u_0, u_0)*dx(degree=4) + 2*L*inner(grad(Q_0), grad(Q_0))*dx(degree=4) + 2*r_0**2*dx(scheme="KMV", degree=1) + (4/3)*dt**2 * inner(grad(p_0), grad(p_0))*dx(degree=4))
record_energy(0.0, E_tot_0, E_disc_0)

t.assign(float(t)+dt) # Advance to t=Δt
def disc_ic_1(): # t=Δt
	# First part
	utildeQHr = Function(FES_hydro)
	u_tilde_1_trial, Q_1_trial, H_1_trial, r_1_trial = split(utildeQHr)

	s1 = S(u_tilde_1_trial, Q_0)
	sigma1 = sigma(Q_0, H_1_trial)

	G1 = (1/dt)*inner(u_tilde_1_trial-u_0, v)*dx(degree=4) + conv_term(u_tilde_0, u_tilde_1_trial, v)*dx(degree=4) + inner(grad(p_0), v)*dx(degree=4) + mu*inner(grad(u_tilde_1_trial), grad(v))*dx(degree=4) \
	 + inner(sigma1, grad(v))*dx(degree=4) \
	 + inner(H_1_trial, contract(grad(Q_0), v))*dx(degree=4)

	G2 = (1/dt)*inner(Q_1_trial-Q_0, Y)*dx(degree=4) + inner(contract(grad(Q_0), u_tilde_1_trial), Y)*dx(degree=4) - inner(s1, Y)*dx(degree=4) \
	 - M*inner(H_1_trial, Y)*dx(degree=4)

	G3 = inner(r_1_trial-r_0, w)*dx(scheme="KMV", degree=1) - inner(inner(P(Q_0), Q_1_trial-Q_0), w)*dx(scheme="KMV", degree=1)

	G4 = inner(H_1_trial, Z)*dx(degree=4) + L*inner(grad(Q_1_trial), grad(Z))*dx(degree=4) + inner(r_1_trial*P(Q_0), Z)*dx(scheme="KMV", degree=1)

	G = G1+G2+G3+G4

	solve(G==0, utildeQHr, bcs=bc_hydro, solver_parameters=solver_parameters)
	u_tilde_1, Q_1, H_1, r_1 = utildeQHr.subfunctions # u1 needs to be computed still

	# Second part
	yp_proj_p = Function(FES_proj)
	y_proj_trial, p_proj_trial, p_trial = split(yp_proj_p)

	F_proj = (1/dt)*inner(y_proj_trial+grad(p_proj_trial)-u_tilde_1, z+grad(hat_q))*dx(degree=4) + inner(grad(p_trial-p_0), z+grad(hat_q))*dx(degree=4) + inner(grad(q), y_proj_trial+grad(p_proj_trial))*dx(degree=4)
	solve(F_proj==0, yp_proj_p, nullspace=nullspace, bcs=bc_proj, solver_parameters=solver_parameters)
	y_proj_1, p_proj_1, p_1 = yp_proj_p.subfunctions

	p_1normalized = p_1 - assemble(p_1*dx(degree=4))/assemble(1*dx(mesh))

	return u_tilde_1, y_proj_1, p_proj_1, Q_1, H_1, r_1, p_1normalized

u_tilde_1, y_proj_1, p_proj_1, Q_1, H_1, r_1, p_1 = disc_ic_1()

######################################### START OF THE BDF2 scheme ####################################################
### Allocate persistent history variables

# 2 steps before
u_tilde_m = Function(U_h).assign(u_tilde_0)
y_proj_m = Function(U_h).assign(y_proj_0)
p_proj_m = Function(P_h).assign(p_proj_0)
Q_m = Function(M_h).assign(Q_0)
H_m = Function(M_h)
r_m = Function(X_h).assign(r_0)
p_m = Function(P_h).assign(p_0)

u_m = y_proj_m + grad(p_proj_m)

# 1 step before
u_tilde_mplus1 = Function(U_h).assign(u_tilde_1)
y_proj_mplus1 = Function(U_h).assign(y_proj_1)
p_proj_mplus1 = Function(P_h).assign(p_proj_1)
Q_mplus1 = Function(M_h, name="Q").assign(Q_1)
H_mplus1 = Function(M_h, name="H").assign(H_1)
r_mplus1 = Function(X_h, name="r").assign(r_1)
p_mplus1 = Function(P_h, name="Pressure").assign(p_1)

u_mplus1 = y_proj_mplus1 + grad(p_proj_mplus1)

# New variable # It doesn't matter how we initialize them
u_tilde_mplus2 = Function(U_h).assign(u_tilde_1)

### Step 1
trial_hydro = TrialFunction(FES_hydro)
u_tilde_trial, Q_trial, H_trial, r_trial = split(trial_hydro)
utildeQHr = Function(FES_hydro) # To store the solution

# Extrapolants
u_hat = extrap(u_tilde_m, u_tilde_mplus1)
Q_hat = extrap(Q_m, Q_mplus1)

s_trial = S(u_tilde_trial, Q_hat)
sig_trial = sigma(Q_hat, H_trial)

F1 = (0.5/dt)*inner((3*u_tilde_trial-4*u_mplus1+u_m), v)*dx(degree=4) + conv_term(u_hat, u_tilde_trial, v)*dx(degree=4) \
 + inner(grad(p_mplus1), v)*dx(degree=4) + mu*inner(grad(u_tilde_trial), grad(v))*dx(degree=4) + inner(sig_trial, grad(v))*dx(degree=4) \
 + inner(H_trial, contract(grad(Q_hat), v))*dx(degree=4)

F2 = (0.5/dt)*inner(3*Q_trial-4*Q_mplus1+Q_m, Y)*dx(degree=4) + inner(contract(grad(Q_hat), u_tilde_trial), Y)*dx(degree=4) \
- inner(s_trial, Y)*dx(degree=4) - M*inner(H_trial, Y)*dx(degree=4)

F3 = inner(3*r_trial-4*r_mplus1+r_m, w)*dx(scheme="KMV", degree=1) - inner(inner(P(Q_hat), 3*Q_trial-4*Q_mplus1+Q_m), w)*dx(scheme="KMV", degree=1)

F4 = inner(H_trial, Z)*dx(degree=4) + L*inner(grad(Q_trial), grad(Z))*dx(degree=4) + inner(r_trial*P(Q_hat), Z)*dx(scheme="KMV", degree=1)

F_step1 = F1+F2+F3+F4

a_step1, L_step1 = lhs(F_step1), rhs(F_step1)
problem_step1 = LinearVariationalProblem(a_step1, L_step1, utildeQHr, bcs=bc_hydro)
solver_step1 = LinearVariationalSolver(problem_step1, solver_parameters=solver_parameters)
# We solve F_step == 0 to get ũᵐ⁺¹, Qᵐ⁺¹, Hᵐ⁺¹, rᵐ⁺¹ in the notation of (3.20) of BEDF1.pdf

### Step 2 (Projection step)
trial_proj = TrialFunction(FES_proj)
y_proj_trial, p_proj_trial, p_trial = split(trial_proj)
yp_proj_p = Function(FES_proj) # To store solution

F_step2 = (1.5/dt)*inner(y_proj_trial+grad(p_proj_trial)-u_tilde_mplus2, z+grad(hat_q))*dx(degree=4) + inner(grad(p_trial-p_mplus1), z+grad(hat_q))*dx(degree=4) + inner(grad(q), y_proj_trial+grad(p_proj_trial))*dx(degree=4)

a_step2, L_step2 = lhs(F_step2), rhs(F_step2)
problem_step2 = LinearVariationalProblem(a_step2, L_step2, yp_proj_p, bcs=bc_proj)
solver_step2 = LinearVariationalSolver(problem_step2, nullspace=nullspace, solver_parameters=solver_parameters)

### Energies

bulk_energy = (0.5*r_mplus1**2 - A0)*dx(degree=4)
kinetic_energy = 0.5*L*inner(grad(Q_mplus1), grad(Q_mplus1))*dx(degree=4)
tot_energy = bulk_energy + kinetic_energy # This is the continuous energy

disc_energy = inner(u_mplus1, u_mplus1)*dx(degree=4) + inner(2*u_mplus1-u_m, 2*u_mplus1-u_m)*dx(degree=4) \
	+ L*inner(grad(Q_mplus1), grad(Q_mplus1))*dx(degree=4) + L*inner(grad(2*Q_mplus1-Q_m), grad(2*Q_mplus1-Q_m))*dx(degree=4) \
	+ r_mplus1**2*dx(scheme="KMV", degree=1) + (2*r_mplus1-r_m)**2*dx(scheme="KMV", degree=1) \
	+ (4/3)*dt**2 * inner(grad(p_mplus1), grad(p_mplus1))*dx(degree=4) # This is the discrete energy

iterate = 0
time_array = np.linspace(0, T, int(T/dt) + 1)
step_idx = 2 # Starting after t=0 and t=dt


# Compute energy at time t = Δt
record_energy(t, assemble(tot_energy), assemble(disc_energy))

while float(t) < T - 1e-8:
	t.assign(time_array[step_idx])
	step_idx += 1

	##### Solve Step 1
	solver_step1.solve()
	u_tilde_new, Q_new, H_new, r_new = utildeQHr.subfunctions

	# Save data for Projection step
	u_tilde_mplus2.assign(u_tilde_new)

	##### Solve Step 2
	solver_step2.solve()
	y_proj_new, p_proj_new, p_new = yp_proj_p.subfunctions

	p_newnormalized = p_new - assemble(p_new*dx(degree=4))/assemble(1*dx(mesh))

	##### Update history
	u_tilde_m.assign(u_tilde_mplus1)
	y_proj_m.assign(y_proj_mplus1)
	p_proj_m.assign(p_proj_mplus1)
	Q_m.assign(Q_mplus1)
	H_m.assign(H_mplus1)
	r_m.assign(r_mplus1)
	p_m.assign(p_mplus1)

	u_tilde_mplus1.assign(u_tilde_new)
	y_proj_mplus1.assign(y_proj_new)
	p_proj_mplus1.assign(p_proj_new)
	Q_mplus1.assign(Q_new)
	H_mplus1.assign(H_new)
	r_mplus1.assign(r_new)
	p_mplus1.assign(p_newnormalized)

	record_energy(t, assemble(tot_energy), assemble(disc_energy))

	# #### Exporting for visualization
	iterate += 1
	if iterate % 400 == 0:
		export_to_pvd(u_mplus1, p_mplus1, Q_mplus1, H_mplus1, r_mplus1, time_val=float(t))

if iterate % 400 != 0:
	export_to_pvd(u_mplus1, p_mplus1, Q_mplus1, H_mplus1, r_mplus1, time_val=float(t))

PETSc.Sys.Print("Finished")