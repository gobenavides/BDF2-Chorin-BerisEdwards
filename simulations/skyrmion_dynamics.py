# 2.5D simulation: skyrmion dynamics

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
L_x = 112
L_y = 56

mesh = PeriodicRectangleMesh(L_x, L_y, L_x, L_y, direction="x")

x, y = SpatialCoordinate(mesh)
n_vec = FacetNormal(mesh)
h_max = mesh.comm.allreduce(mesh.cell_sizes.dat.data.max(), op=MPI.MAX)

dim_u = 2 # Spatial velocity dimension
dim_Q = 3 # Tensor order parameter dimension

### Physical parameters
a = 0.00256*2
b = 0.208896*3
c = 0.1152*4
A0 = 500

mu = 0.909
xi = 0.82
M = 0.474
L = 1.0226e-6

### Numerical parameters
T = 600             # End time
dt = 0.1
t = Constant(0.0)   # Current time

# Weak anchoring surface energy strength
W_surface = Constant(1e-3)

################################## Initialize CSV for Energy
csv_filename = "energies.csv"
if mesh.comm.rank == 0:
	with open(csv_filename, mode='w', newline='') as file:
		writer = csv.writer(file, delimiter='\t')
		writer.writerow(["Time", "Continuous_Energy", "Discrete_Energy"])

# Function to export to CSV
def record_energy(time_val, E_c, E_d):
	PETSc.Sys.Print(f"Time {float(time_val):10.8f}, \tContinuous energy: {E_c:10.16f}, \tDiscrete energy: {E_d:10.16f}")
	if mesh.comm.rank == 0:
		with open(csv_filename, mode='a', newline='') as file:
			writer = csv.writer(file, delimiter='\t')
			writer.writerow([float(time_val), E_c, E_d])

#################################### Operators ####################################

def aux_var(A):
	return sqrt(2*( (a/2)*tr(dot(A,A)) -(b/3)*tr( dot(A,dot(A,A)) ) + (c/4)*tr(dot(A,A))**2 + A0 ) )

def sigma(Q,H):
	return dot(Q,H) - dot(H,Q) - xi*(dot(H,Q) + dot(Q,H)) - 2*xi/dim_Q*H + 2*xi*inner(Q,H)*Q

def sym_grad(u):
	return 0.5*(grad(u) + grad(u).T)

def skew_grad(u):
	return 0.5*(grad(u) - grad(u).T)

def pad_2D_to_3D(A_2d):
	"""Pads a 2x2 UFL tensor to 3x3 to interact with 3D Q-tensor"""
	return as_tensor([
		[A_2d[0,0], A_2d[0,1], 0.0],
		[A_2d[1,0], A_2d[1,1], 0.0],
		[0.0,       0.0,       0.0]
	])

def S(u,Q): # This is little s
	# Pad 2D velocity gradients to 3D for coupling
	DD = pad_2D_to_3D(sym_grad(u))
	WW = pad_2D_to_3D(skew_grad(u))

	return dot(WW,Q) - dot(Q,WW) + xi*(dot(Q,DD) + dot(DD,Q)) \
	+ (2*xi/dim_Q)*DD - (2*xi/dim_Q**2)*div(u)*Identity(dim_Q) - 2*xi*inner(DD,Q)*(Q + (1/dim_Q)*Identity(dim_Q))

def P(Q):
	V = a*Q - b*(dot(Q,Q) - (1/dim_Q)*tr(dot(Q,Q))*Identity(dim_Q)) + c*tr(dot(Q,Q))*Q
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

#################################### Finite Element spaces ####################################

# Taylor-Hood pair
U_h = VectorFunctionSpace(mesh, "CG", 2, dim=dim_u) # Intermediate velocity space
P_h = FunctionSpace(mesh, "CG", 1)              # Pressure space, we should enforce mean-zero!

X_h = FunctionSpace(mesh, "CG", 1)              # Auxiliary variable space
# Space for 3x3 Q-tensor
M_h = TensorFunctionSpace(mesh, "CG", 1, shape=(dim_Q, dim_Q), symmetry=symmetry)

# FE Space for the hydrodynamics
FES_hydro = U_h * M_h * M_h * X_h

### Problem data
epsilon = Constant(1e-10) # To avoid division by zero

# Initial and boundary conditions
m = 1
g = pi/2
R = 0.7*16
B = 0.5
C_x = L_x/2
C_y = L_y/2

tilde_b = atan2(x-C_x, y-C_y)
rho = sqrt((x-C_x)**2 + (y-C_y)**2)
tilde_a = (pi/2)*(1-tanh(0.5*B*(rho-R)))

n_x = sin(tilde_a)*sin(m*tilde_b+g)
n_y = sin(tilde_a)*cos(m*tilde_b+g)
n_z = -cos(tilde_a)

n_in = as_vector([n_x, n_y, n_z]) # Initial director field is natively 3D
Q_in = outer(n_in,n_in) - (1/dim_Q)*Identity(dim_Q) # Traceless symmetric 3x3
r_in = aux_var(Q_in)


bc_u_updown = DirichletBC(FES_hydro.sub(0), as_vector([0, 0]), (1,2))
u_avg = 1
u_max = u_avg*1.5
beta = 8*mu*u_max/(L_y**2)
f_poiseuille = beta*as_vector([1,0])

# Define constant homeotropic director
n_wall = as_vector([0, 0, 1]) 
Q_wall = outer(n_wall, n_wall) - (1/dim_Q)*Identity(dim_Q)

# Neumann condition for Q on top and bottom, meaning NO DirichletBC for Q
bc_hydro = [bc_u_updown] 

u_in = as_vector([x-x,0])

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

# Define NEW boundary conditions exclusively for FES_proj
bc_proj_updown = DirichletBC(FES_proj.sub(0), as_vector([0, 0]), (1,2))
bc_proj = [bc_proj_updown]

z, hat_q, q = TestFunctions(FES_proj) # Some test functions. We will use them many times

################################### For visualization ##############################################
outfile = VTKFile("skyrmion_periodic_leftright_weakanchoring.pvd")

# Dedicated spaces and persistent functions for ParaView visualization
U_out_space = VectorFunctionSpace(mesh, "CG", 2, dim=dim_u)
u_out = Function(U_out_space, name="Velocity")
p_out = Function(P_h, name="Pressure")
Q_out = Function(M_h, name="Q")
H_out = Function(M_h, name="H")
r_out = Function(X_h, name="r")

S_out_space = FunctionSpace(mesh, "CG", 1)
S_out = Function(S_out_space, name="Order_Parameter")

nc_out_space = FunctionSpace(mesh, "CG", 3)
nc_out = Function(nc_out_space, name="Non-conformity parameter")

kk = as_vector([0,0,1]) # out of plane normal
PP = Identity(3) - outer(kk,kk) # pointwise tangential projection

def compute_noncorf(AA):
	"""
	Computes the non-conformity parameter of the 3x3 Q-tensor.
	"""
	n_hom = inner(kk, dot(AA, kk)) 
	N_hom = n_hom * (outer(kk, kk) - 0.5 * PP)
	
	T_hom = dot(PP, dot(AA, PP)) + 0.5 * n_hom * PP
	
	R_hom = AA - T_hom - N_hom
	
	return conditional(le(inner(AA, AA),1e-14), 0.0, sqrt(inner(R_hom, R_hom) / (inner(AA, AA))))

def export_to_pvd(u_val, p_val, Q_val, H_val, r_val, S_expr, nc_expr, time_val):
	"""Helper to safely interpolate expressions and write native functions directly to VTK."""

	u_out.interpolate(u_val)
	p_out.interpolate(p_val)
	Q_out.assign(Q_val)
	H_out.assign(H_val)
	r_out.assign(r_val)

	S_out.interpolate(S_expr)
	nc_out.interpolate(nc_expr)

	outfile.write(u_out, p_out, Q_out, H_out, r_out, S_out, nc_out, time=float(time_val))
	PETSc.Sys.Print(f"PVD saved")

#################################### Discrete initial conditions ####################################
def disc_ic_0(): # t = 0
	u_tilde_0 = project(u_in, U_h)

	yp_proj_p = Function(FES_proj)
	y_proj_trial, p_proj_trial, p_trial = split(yp_proj_p)

	F_proj = (1/dt)*inner(y_proj_trial+grad(p_proj_trial)-u_tilde_0, z+grad(hat_q))*dx(degree=4) + inner(grad(p_trial), z+grad(hat_q))*dx(degree=4) + inner(grad(q), y_proj_trial+grad(p_proj_trial))*dx(degree=4)
	solve(F_proj==0, yp_proj_p, nullspace=nullspace, bcs=bc_proj, solver_parameters=solver_parameters)

	y_proj_0, p_proj_0, p_0 = yp_proj_p.subfunctions

	p_0normalized = p_0 - assemble(p_0*dx(degree=4))/assemble(1*dx(mesh))

	Q_0 = Function(M_h).interpolate(Q_in)

	r_0 = Function(X_h)
	r_0.interpolate(r_in)

	return u_tilde_0, y_proj_0, p_proj_0, Q_0, r_0, p_0normalized

u_tilde_0, y_proj_0, p_proj_0, Q_0, r_0, p_0 = disc_ic_0()
u_0 = y_proj_0 + grad(p_proj_0)

# --- Export initial state (t = 0) ---
H_0 = Function(M_h, name="H")

S_expr_0 = 2 * sqrt(Q_0[0, 0]**2 + Q_0[0, 1]**2)
nc_expr_0 = compute_noncorf(Q_0)

export_to_pvd(u_0, p_0, Q_0, H_0, r_0, S_expr_0, nc_expr_0, time_val=0.0)

# --- Compute and log t=0 energy ---
# Modified to include the surface anchoring energy
E_tot_0 = assemble((0.5*r_0**2 - A0)*dx(degree=4) + 0.5*L*inner(grad(Q_0), grad(Q_0))*dx(degree=4) + 0.5*W_surface*inner(Q_0 - Q_wall, Q_0 - Q_wall)*ds)
E_disc_0 = assemble(2*inner(u_0, u_0)*dx(degree=4) + 2*L*inner(grad(Q_0), grad(Q_0))*dx(degree=4) + 2*r_0**2*dx(scheme="KMV", degree=1) + (4/3)*dt**2 * inner(grad(p_0), grad(p_0))*dx(degree=4) + 2*W_surface*inner(Q_0 - Q_wall, Q_0 - Q_wall)*ds)
record_energy(0.0, E_tot_0, E_disc_0)

t.assign(float(t)+dt) # Advance to t=Δt
def disc_ic_1(): # t=Δt
	utildeQHr = Function(FES_hydro)
	u_tilde_1_trial, Q_1_trial, H_1_trial, r_1_trial = split(utildeQHr)

	s1 = S(u_tilde_1_trial, Q_0)
	sigma1 = sigma(Q_0, H_1_trial)

	G1 = (1/dt)*inner(u_tilde_1_trial-u_0, v)*dx(degree=4) + conv_term(u_tilde_0, u_tilde_1_trial, v)*dx(degree=4) + inner(grad(p_0), v)*dx(degree=4) + mu*inner(grad(u_tilde_1_trial), grad(v))*dx(degree=4) \
	 + inner(sigma1, pad_2D_to_3D(grad(v)))*dx(degree=4) \
	 + inner(H_1_trial, contract(grad(Q_0), v))*dx(degree=4) \
	 - inner(f_poiseuille, v)*dx(degree=4)

	G2 = (1/dt)*inner(Q_1_trial-Q_0, Y)*dx(degree=4) + inner(contract(grad(Q_0), u_tilde_1_trial), Y)*dx(degree=4) - inner(s1, Y)*dx(degree=4) \
	 - M*inner(H_1_trial, Y)*dx(degree=4)

	G3 = inner(r_1_trial-r_0, w)*dx(scheme="KMV", degree=1) - inner(inner(P(Q_0), Q_1_trial-Q_0), w)*dx(scheme="KMV", degree=1)

	# Modified G4: Added the surface anchoring term (*ds)
	G4 = inner(H_1_trial, Z)*dx(degree=4) + L*inner(grad(Q_1_trial), grad(Z))*dx(degree=4) + inner(r_1_trial*P(Q_0), Z)*dx(scheme="KMV", degree=1) \
	   + W_surface * inner(Q_1_trial - Q_wall, Z) * ds

	G = G1+G2+G3+G4

	solve(G==0, utildeQHr, bcs=bc_hydro, solver_parameters=solver_parameters)
	u_tilde_1, Q_1, H_1, r_1 = utildeQHr.subfunctions

	yp_proj_p = Function(FES_proj)
	y_proj_trial, p_proj_trial, p_trial = split(yp_proj_p)

	F_proj = (1/dt)*inner(y_proj_trial+grad(p_proj_trial)-u_tilde_1, z+grad(hat_q))*dx(degree=4) + inner(grad(p_trial-p_0), z+grad(hat_q))*dx(degree=4) + inner(grad(q), y_proj_trial+grad(p_proj_trial))*dx(degree=4)
	solve(F_proj==0, yp_proj_p, nullspace=nullspace, bcs=bc_proj, solver_parameters=solver_parameters)
	y_proj_1, p_proj_1, p_1 = yp_proj_p.subfunctions

	p_1normalized = p_1 - assemble(p_1*dx(degree=4))/assemble(1*dx(mesh))

	return u_tilde_1, y_proj_1, p_proj_1, Q_1, H_1, r_1, p_1normalized

u_tilde_1, y_proj_1, p_proj_1, Q_1, H_1, r_1, p_1 = disc_ic_1()

######################################### START OF THE BDF2 scheme ####################################################

u_tilde_m = Function(U_h).assign(u_tilde_0)
y_proj_m = Function(U_h).assign(y_proj_0)
p_proj_m = Function(P_h).assign(p_proj_0)
Q_m = Function(M_h).assign(Q_0)
H_m = Function(M_h)
r_m = Function(X_h).assign(r_0)
p_m = Function(P_h).assign(p_0)

u_m = y_proj_m + grad(p_proj_m)

u_tilde_mplus1 = Function(U_h).assign(u_tilde_1)
y_proj_mplus1 = Function(U_h).assign(y_proj_1)
p_proj_mplus1 = Function(P_h).assign(p_proj_1)
Q_mplus1 = Function(M_h, name="Q").assign(Q_1)
H_mplus1 = Function(M_h, name="H").assign(H_1)
r_mplus1 = Function(X_h, name="r").assign(r_1)
p_mplus1 = Function(P_h, name="Pressure").assign(p_1)

u_mplus1 = y_proj_mplus1 + grad(p_proj_mplus1)

u_tilde_mplus2 = Function(U_h).assign(u_tilde_1)

### Step 1
trial_hydro = TrialFunction(FES_hydro)
u_tilde_trial, Q_trial, H_trial, r_trial = split(trial_hydro)
utildeQHr = Function(FES_hydro) 

u_hat = extrap(u_tilde_m, u_tilde_mplus1)
Q_hat = extrap(Q_m, Q_mplus1)

s_trial = S(u_tilde_trial, Q_hat)
sig_trial = sigma(Q_hat, H_trial)

F1 = (0.5/dt)*inner((3*u_tilde_trial-4*u_mplus1+u_m), v)*dx(degree=4) + conv_term(u_hat, u_tilde_trial, v)*dx(degree=4) \
 + inner(grad(p_mplus1), v)*dx(degree=4) + mu*inner(grad(u_tilde_trial), grad(v))*dx(degree=4) + inner(sig_trial, pad_2D_to_3D(grad(v)))*dx(degree=4) \
 + inner(H_trial, contract(grad(Q_hat), v))*dx(degree=4) \
 - inner(f_poiseuille, v)*dx(degree=4)

F2 = (0.5/dt)*inner(3*Q_trial-4*Q_mplus1+Q_m, Y)*dx(degree=4) + inner(contract(grad(Q_hat), u_tilde_trial), Y)*dx(degree=4) \
- inner(s_trial, Y)*dx(degree=4) - M*inner(H_trial, Y)*dx(degree=4)

F3 = inner(3*r_trial-4*r_mplus1+r_m, w)*dx(scheme="KMV", degree=1) - inner(inner(P(Q_hat), 3*Q_trial-4*Q_mplus1+Q_m), w)*dx(scheme="KMV", degree=1)

# Modified F4: Added the surface anchoring term (*ds)
F4 = inner(H_trial, Z)*dx(degree=4) + L*inner(grad(Q_trial), grad(Z))*dx(degree=4) + inner(r_trial*P(Q_hat), Z)*dx(scheme="KMV", degree=1) \
   + W_surface * inner(Q_trial - Q_wall, Z) * ds

F_step1 = F1+F2+F3+F4

a_step1, L_step1 = lhs(F_step1), rhs(F_step1)
problem_step1 = LinearVariationalProblem(a_step1, L_step1, utildeQHr, bcs=bc_hydro)
solver_step1 = LinearVariationalSolver(problem_step1, solver_parameters=solver_parameters)

### Step 2 (Projection step)
trial_proj = TrialFunction(FES_proj)
y_proj_trial, p_proj_trial, p_trial = split(trial_proj)
yp_proj_p = Function(FES_proj) 

F_step2 = (1.5/dt)*inner(y_proj_trial+grad(p_proj_trial)-u_tilde_mplus2, z+grad(hat_q))*dx(degree=4) + inner(grad(p_trial-p_mplus1), z+grad(hat_q))*dx(degree=4) + inner(grad(q), y_proj_trial+grad(p_proj_trial))*dx(degree=4)

a_step2, L_step2 = lhs(F_step2), rhs(F_step2)
problem_step2 = LinearVariationalProblem(a_step2, L_step2, yp_proj_p, bcs=bc_proj)
solver_step2 = LinearVariationalSolver(problem_step2, nullspace=nullspace, solver_parameters=solver_parameters)

### Energies
# Modified to include the surface anchoring energy terms
bulk_energy = (0.5*r_mplus1**2 - A0)*dx(degree=4)
kinetic_energy = 0.5*L*inner(grad(Q_mplus1), grad(Q_mplus1))*dx(degree=4)
anchoring_energy = 0.5*W_surface*inner(Q_mplus1 - Q_wall, Q_mplus1 - Q_wall)*ds
tot_energy = bulk_energy + kinetic_energy + anchoring_energy # Continuous energy

disc_energy = inner(u_mplus1, u_mplus1)*dx(degree=4) + inner(2*u_mplus1-u_m, 2*u_mplus1-u_m)*dx(degree=4) \
	+ L*inner(grad(Q_mplus1), grad(Q_mplus1))*dx(degree=4) + L*inner(grad(2*Q_mplus1-Q_m), grad(2*Q_mplus1-Q_m))*dx(degree=4) \
	+ r_mplus1**2*dx(scheme="KMV", degree=1) + (2*r_mplus1-r_m)**2*dx(scheme="KMV", degree=1) \
	+ (4/3)*dt**2 * inner(grad(p_mplus1), grad(p_mplus1))*dx(degree=4) \
    + W_surface*inner(Q_mplus1 - Q_wall, Q_mplus1 - Q_wall)*ds + W_surface*inner((2*Q_mplus1 - Q_m) - Q_wall, (2*Q_mplus1 - Q_m) - Q_wall)*ds

iterate = 1
time_array = np.linspace(0, T, int(T/dt) + 1)
step_idx = 2 

record_energy(t, assemble(tot_energy), assemble(disc_energy))

while float(t) < T - 1e-8:
	t.assign(time_array[step_idx])
	step_idx += 1

	##### Solve Step 1
	solver_step1.solve()
	u_tilde_new, Q_new, H_new, r_new = utildeQHr.subfunctions

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

	iterate += 1
	if iterate % 100 == 0:
		S_expr_mplus1 = 2 * sqrt(Q_mplus1[0, 0]**2 + Q_mplus1[0, 1]**2)
		nc_expr_mplus1 = compute_noncorf(Q_mplus1)

		export_to_pvd(u_mplus1, p_mplus1, Q_mplus1, H_mplus1, r_mplus1, S_expr_mplus1, nc_expr_mplus1, time_val=float(t))

if iterate % 100 != 0:
	S_expr_mplus1 = 2 * sqrt(Q_mplus1[0, 0]**2 + Q_mplus1[0, 1]**2)
	nc_expr_mplus1 = compute_noncorf(Q_mplus1)

	export_to_pvd(u_mplus1, p_mplus1, Q_mplus1, H_mplus1, r_mplus1, S_expr_mplus1, nc_expr_mplus1, time_val=float(t))

PETSc.Sys.Print("Finished")