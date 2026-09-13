# Temporal convergence test

from firedrake import *
import numpy as np
import finat
from mpi4py import MPI

# For some reason I don't understand I need to enforce boundary conditions on H too! Otherwise the scheme doesn't converge.

#################################### Mesh, geometric and physical quantities ####################################

N = 6 # Number of refinements
symmetry = True # If we want the space for Q and H to be strongly symmetric.

solver_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
}

def solve_BerisEdwards(nn):
	PETSc.Sys.Print(f"Starting Iteration {nn} out of {N}")

	mesh = SquareMesh(2**7, 2**7, 2) # [0,2]^2
	h_max = mesh.comm.allreduce(mesh.cell_sizes.dat.data.max(), op=MPI.MAX)
	x, y = SpatialCoordinate(mesh)
	n_vec = FacetNormal(mesh)

	d = 2 # Space dimension

	### Physical parameters
	a = 1
	b = 0
	c = 1
	A0 = 1

	mu = 1
	xi = 1
	M = 1
	L = 1

	### Numerical parameters
	T = 1             # End time
	
	n_timepoints = 5*(nn+1)
	# n_timepoints = 10 * 2**(nn-1) + 1
	t_linspace = list(np.linspace(0,T,num=n_timepoints))

	dt = T/(n_timepoints-1)
	t = Constant(t_linspace.pop(0))               # Current time

	#################################### Operators ####################################

	def aux_var(A):
		return sqrt(2*( (a/2)*tr(dot(A,A)) -(b/3)*tr( dot(A,dot(A,A)) ) + (c/4)*tr(dot(A,A))**2 + A0 ) )

	def sigma(Q,H):
		return dot(Q,H) - dot(H,Q) - xi*(dot(H,Q) + dot(Q,H)) - 2*xi/d*H + 2*xi*inner(Q,H)*Q

	def sym_grad(u):
		return 0.5*(grad(u) + grad(u).T)

	def skew_grad(u):
		return 0.5*(grad(u) - grad(u).T)

	def S(u,Q): #This is little s
		DD = sym_grad(u)
		WW = skew_grad(u)

		return dot(WW,Q) - dot(Q,WW) + xi*(dot(Q,DD) + dot(DD,Q)) \
		+ (2*xi/d)*DD - (2*xi/d**2)*div(u)*Identity(d) - 2*xi*inner(DD,Q)*(Q + (1/d)*Identity(d))

	def P(Q):
		V = a*Q - b*(dot(Q,Q) - (1/d)*tr(dot(Q,Q))*Identity(d)) + c*tr(dot(Q,Q))*Q
		return V/aux_var(Q)

	def conv_term(uu,vv,ww):
		return (inner(dot(grad(vv),uu),ww) + 0.5*div(uu)*inner(vv,ww))

	def extrap(A0,A1):
		return 2*A1 - A0

	def contract(RR,vv):
		i,j,k = indices(3)
		return as_tensor( RR[i,j,k]*vv[k], (i,j))

	def contract_higher(AA,RR):
		i,j,k = indices(3)
		return as_vector( AA[i,j]*RR[i,j,k], k)

	def trac_sym(AA):
		# Given a 2nd order tensor, it returns its pointwise projection onto the space of symmetric and traceless tensors
		BB = 0.5*(AA+AA.T)
		return BB - 0.5*tr(BB)*Identity(2)

	##3 Exact solutions
	t_var = variable(t)

	u_ex = exp(-t_var)*as_vector([pi *(1-cos(pi*x))*sin(pi*y),-pi*sin(pi*x)*(1 - cos(pi*y))])
	n_ex = as_vector([cos(pi*x*t_var)*cos(0.5*pi*y*t_var), sin(pi*x*t_var)*sin(0.5*pi*y*t_var)])
	Q_ex = outer(n_ex,n_ex) - 0.5*inner(n_ex,n_ex)*Identity(2) # Traceless symmetric
	r_ex = aux_var(Q_ex)
	p_ex = (x-1)**3 * sin(2*pi*y*t_var)
	# p_ex = cos(pi*x) * cos(pi*y) * sin(t_var) # This ensures the pressure satisfies 0 Neumann conditions.
	# H_ex = L*div(grad(Q_ex)) - r_ex*P(Q_ex)
	if symmetry:
		H_ex = trac_sym( L*div(grad(Q_ex)) - r_ex*P(Q_ex) ) #Ensures that H_ex is symmetric and traceless
	else:
		H_ex = L*div(grad(Q_ex)) - r_ex*P(Q_ex)

	ff = diff(u_ex,t_var) + dot(grad(u_ex),u_ex) + grad(p_ex) - mu*div(grad(u_ex)) - div(sigma(Q_ex,H_ex)) + contract_higher(H_ex,grad(Q_ex)) #RHS of the momentum equation
	GG = diff(Q_ex,t_var) + contract(grad(Q_ex),u_ex) - S(u_ex,Q_ex) - M*H_ex # This is the RHS that we add to the kinematic equation.

	#################################### Finite Element spaces ####################################

	# Taylor-Hood pair
	U_h = VectorFunctionSpace(mesh,"CG",2, dim=2) #Intermediate velocity space # Don't forget about the bcs!
	P_h = FunctionSpace(mesh,"CG",1) #Pressure space, we should enforce mean-zero!
	# R = FunctionSpace(mesh,"R",0) # Real space for the real Lagrange multiplier

	X_h = FunctionSpace(mesh,"CG",1) #Auxiliary variable space
	M_h = TensorFunctionSpace(mesh,"CG",1, symmetry=symmetry) #Space for Q-tensor (should I assume it is symmetric?)
	# Y_h = U_h ⊕ ∇P_h. Each function in Y_h can be uniquely decomposed!

	### Mixed FE

	# Space for the hydrodynamics
	FES_hydro = U_h * M_h * M_h * X_h
	bc_u = DirichletBC(FES_hydro.sub(0), u_ex, (1,2,3,4))
	bc_Q = DirichletBC(FES_hydro.sub(1), Q_ex, (1,2,3,4))
	# bc_H = DirichletBC(FES_hydro.sub(2), H_ex, (1,2,3,4))
	# bc_hydro = [bc_u, bc_Q, bc_H]
	bc_hydro = [bc_u, bc_Q]

	v,Z,Y,w = TestFunctions(FES_hydro) #Some test functions. We will use them many times

	# Space for the projection steps
	FES_proj = U_h * P_h * P_h
	nullspace = MixedVectorSpaceBasis(
	    FES_proj, [FES_proj.sub(0), VectorSpaceBasis(constant=True), VectorSpaceBasis(constant=True)]
	) # To remove constants in the pressure space
	bc_proj = DirichletBC(FES_proj.sub(0), u_ex, (1,2,3,4)) #Dirichlet boundary condition

	z, hat_q, q = TestFunctions(FES_proj) #Some test functions. We will use them many times

	#################################### Discrete initial conditions ####################################
	def disc_ic_0(): #t = 0
		#L2 Projection of u_ex(t=0) on U_h
		u_tilde_0 = project(u_ex,U_h, bcs=DirichletBC(U_h, u_ex, (1,2,3,4)) ) # I had forgotten about the boundary conditions

		#Projection of u_tilde_0 in the "weakly" divergence functions of V_h.
		yp_proj_p = Function(FES_proj)
		y_proj_trial, p_proj_trial, p_trial = split(yp_proj_p)

		F_proj = (1/dt)*inner(y_proj_trial+grad(p_proj_trial)-u_tilde_0, z+grad(hat_q))*dx(degree=4) + inner(grad(p_trial), z+grad(hat_q))*dx(degree=4) + inner(grad(q), y_proj_trial+grad(p_proj_trial))*dx(degree=4)
		solve(F_proj==0, yp_proj_p, nullspace=nullspace, bcs=bc_proj, solver_parameters=solver_parameters)

		y_proj_0, p_proj_0, p_0 = yp_proj_p.subfunctions #We care about u_h = y_h + ∇p̂ₕ, and the pressure pₕ

		p_0normalized = p_0 - assemble(p_0*dx(degree=4))/assemble(1*dx(mesh))

		#L2 Projection of Q_0 in M_h
		Q_0 = project(Q_ex,M_h,bcs=DirichletBC(M_h, Q_ex, (1,2,3,4))) # I had forgotten about the boundary conditions

		#Lagrage interpolant of r(Q_0)
		r_0 = Function(X_h)
		r_0.interpolate(r_ex)

		return u_tilde_0, y_proj_0, p_proj_0, Q_0, r_0, p_0normalized

	u_tilde_0, y_proj_0, p_proj_0, Q_0, r_0, p_0 = disc_ic_0()
	u_0 = y_proj_0 + grad(p_proj_0) # This is a temporary definition, we will delete it afterwards

	t.assign(t_linspace.pop(0)) #Advance to t=Δt
	def disc_ic_1(): #t=Δt
		# First part
		utildeQHr = Function(FES_hydro)
		u_tilde_1_trial, Q_1_trial, H_1_trial, r_1_trial = split(utildeQHr)

		s1 = S(u_tilde_1_trial,Q_0)
		sigma1 = sigma(Q_0,H_1_trial)

		G1 = (1/dt)*inner(u_tilde_1_trial-u_0,v)*dx(degree=4) + conv_term(u_tilde_0,u_tilde_1_trial,v)*dx(degree=4) + inner(grad(p_0),v)*dx(degree=4) + mu*inner(grad(u_tilde_1_trial),grad(v))*dx(degree=4) \
		 + inner(sigma1,grad(v))*dx(degree=4) \
		 + inner(H_1_trial, contract(grad(Q_0),v) )*dx(degree=4) - inner(ff,v)*dx(degree=4)

		G2 = (1/dt)*inner(Q_1_trial-Q_0,Y)*dx(degree=4) + inner( contract(grad(Q_0), u_tilde_1_trial) , Y)*dx(degree=4) - inner(s1,Y)*dx(degree=4) \
		 - M*inner(H_1_trial,Y)*dx(degree=4) - inner(GG,Y)*dx(degree=4)
		 # the term inner(GG,Y)*dx(degree=4) makes up for the RHS function GG in the kinematic equation of LC

		G3 = inner(r_1_trial-r_0,w)*dx(scheme="KMV", degree=1) - inner(inner(P(Q_0),Q_1_trial-Q_0),w)*dx(scheme="KMV", degree=1)

		G4 = inner(H_1_trial,Z)*dx(degree=4) + L*inner(grad(Q_1_trial),grad(Z))*dx(degree=4) + inner(r_1_trial*P(Q_0),Z)*dx(scheme="KMV", degree=1) \
		 # - L*inner(dot(grad(Q_ex), n_vec), Z)*ds(degree=4) # I was forgetting a boundary term involving the normal derivative of Q

		G = G1+G2+G3+G4

		solve(G==0, utildeQHr, bcs=bc_hydro, solver_parameters=solver_parameters)
		u_tilde_1, Q_1, H_1, r_1 = utildeQHr.subfunctions #u1 needs to be computed still

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
	H_m = Function(M_h)#.assign(H_0)
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
	p_mplus1 = Function(P_h, name="p").assign(p_1)

	u_mplus1 = y_proj_mplus1 + grad(p_proj_mplus1)

	# New variable # It doesn't matter how we initialize them
	u_tilde_mplus2 = Function(U_h).assign(u_tilde_1)

	### Step 1

	utildeQHr = Function(FES_hydro)
	u_tilde_trial, Q_trial, H_trial, r_trial = split(utildeQHr)

	# Extrapolants
	u_hat = extrap(u_tilde_m,u_tilde_mplus1)
	Q_hat = extrap(Q_m,Q_mplus1)

	s_trial = S(u_tilde_trial,Q_hat)
	sig_trial = sigma(Q_hat,H_trial)

	F1 = (0.5/dt)*inner((3*u_tilde_trial-4*u_mplus1+u_m), v)*dx(degree=4) + conv_term(u_hat, u_tilde_trial, v)*dx(degree=4) \
	 + inner(grad(p_mplus1),v)*dx(degree=4) + mu*inner(grad(u_tilde_trial),grad(v))*dx(degree=4) + inner(sig_trial,grad(v))*dx(degree=4) \
	 + inner(H_trial, contract(grad(Q_hat),v) )*dx(degree=4) - inner(ff,v)*dx(degree=4)

	F2 = (0.5/dt)*inner(3*Q_trial-4*Q_mplus1+Q_m, Y)*dx(degree=4) + inner( contract(grad(Q_hat), u_tilde_trial) , Y)*dx(degree=4) \
	 - inner(s_trial,Y)*dx(degree=4) - M*inner(H_trial,Y)*dx(degree=4) - inner(GG,Y)*dx(degree=4)
	# the term inner(GG,Y)*dx(degree=4) makes up for the RHS function GG in the kinematic equation of LC

	F3 = inner(3*r_trial-4*r_mplus1+r_m,w)*dx(scheme="KMV", degree=1) - inner(inner(P(Q_hat),3*Q_trial-4*Q_mplus1+Q_m),w)*dx(scheme="KMV", degree=1)

	F4 = inner(H_trial,Z)*dx(degree=4) + L*inner(grad(Q_trial),grad(Z))*dx(degree=4) + inner(r_trial*P(Q_hat), Z)*dx(scheme="KMV", degree=1) \
	 # - L*inner(dot(grad(Q_ex), n_vec), Z)*ds(degree=4) # I was forgetting a boundary term involving the normal derivative of Q

	F_step1 = F1+F2+F3+F4

	problem_step1 = NonlinearVariationalProblem(F_step1, utildeQHr, bcs=bc_hydro)
	solver_step1 = NonlinearVariationalSolver(problem_step1, solver_parameters=solver_parameters)
	# We solve F_step == 0 to get ũᵐ⁺¹, Qᵐ⁺¹, Hᵐ⁺¹, rᵐ⁺¹ in the notation of (3.20) of BEDF1.pdf

	### Step 2 (Projection step)

	yp_proj_p = Function(FES_proj)
	y_proj_trial, p_proj_trial, p_trial = split(yp_proj_p)

	F_step2 = (1.5/dt)*inner(y_proj_trial+grad(p_proj_trial)-u_tilde_mplus2, z+grad(hat_q))*dx(degree=4) + inner(grad(p_trial-p_mplus1), z+grad(hat_q))*dx(degree=4) + inner(grad(q), y_proj_trial+grad(p_proj_trial))*dx(degree=4)
	problem_step2 = NonlinearVariationalProblem(F_step2, yp_proj_p, bcs=bc_proj)
	solver_step2 = NonlinearVariationalSolver(problem_step2, nullspace=nullspace, solver_parameters=solver_parameters)

	### For visualization
	# outfile = VTKFile("Beris-Edwards.pvd")
	# P1 = VectorFunctionSpace(mesh, "CG", 1) #Just for exporting
	# u_output = Function(P1, name="u")

	while float(t) < T - 1e-8:
		t.assign(t_linspace.pop(0))

		#####Solve Step 1
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

		##### Exporting for visualization
		# outfile.write(u_output, Q_mplus1, H_mplus1, r_mplus1, p_mplus1, time=t)
	
	PETSc.Sys.Print(f"Computing error at final time t={float(t)}")
	L2error_u = sqrt(assemble( inner(u_ex-u_mplus1,u_ex-u_mplus1) *dx(degree=4)))
	L2error_Q = sqrt(assemble( inner(Q_ex-Q_mplus1,Q_ex-Q_mplus1) *dx(degree=4)))
	L2error_H = sqrt(assemble( inner(H_ex-H_mplus1,H_ex-H_mplus1) *dx(degree=4)))
	L2error_r = sqrt(assemble( inner(r_ex-r_mplus1,r_ex-r_mplus1) *dx(degree=4)))

	p_mplus1_meanzero = p_mplus1 - assemble(p_mplus1*dx(degree=4))/assemble(1*dx(mesh))
	L2error_p = sqrt(assemble( inner(p_ex-p_mplus1_meanzero,p_ex-p_mplus1_meanzero) *dx(degree=4)))

	return dt, L2error_u, L2error_Q, L2error_H, L2error_r, L2error_p

results = [solve_BerisEdwards(nn) for nn in range(1,N+1)]
PETSc.Sys.Print("Finished\n")

PETSc.Sys.Print(f"\n{'Δt':>10} | {'u_err':>10} | {'u_rate':>8} | {'Q_err':>10} | {'Q_rate':>8} | {'H_err':>10} | {'H_rate':>8} | {'r_err':>10} | {'r_rate':>8} | {'p_err':>10} | {'p_rate':>8}")
PETSc.Sys.Print("-" * 131)

for i in range(len(results)):
	dt, u_e, Q_e, H_e, r_e, p_e = results[i]
	if i == 0:
		PETSc.Sys.Print(f"{dt:10.4f} | {u_e:10.4e} | {'-':>8} | {Q_e:10.4e} | {'-':>8} | {H_e:10.4e} | {'-':>8} | {r_e:10.4e} | {'-':>8} | {p_e:10.4e} | {'-':>8}")
	else:
		dt_prev, u_e_prev, Q_e_prev, H_e_prev, r_e_prev, p_e_prev = results[i-1]
		u_rate = np.log(u_e / u_e_prev) / np.log(dt / dt_prev)
		Q_rate = np.log(Q_e / Q_e_prev) / np.log(dt / dt_prev)
		H_rate = np.log(H_e / H_e_prev) / np.log(dt / dt_prev)
		r_rate = np.log(r_e / r_e_prev) / np.log(dt / dt_prev)
		p_rate = np.log(p_e / p_e_prev) / np.log(dt / dt_prev)
		PETSc.Sys.Print(f"{dt:10.4f} | {u_e:10.4e} | {u_rate:8.2f} | {Q_e:10.4e} | {Q_rate:8.2f} | {H_e:10.4e} | {H_rate:8.2f} | {r_e:10.4e} | {r_rate:8.2f} | {p_e:10.4e} | {p_rate:8.2f}")