import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import default_scalar_type, geometry, fem
from dolfinx.common import Timer, list_timings
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem import functionspace, Constant, dirichletbc, locate_dofs_topological, assemble_scalar,form, Function
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import dx, grad, inner, ds, Measure
import dolfinx_mpc.utils
# from dolfinx_mpc import LinearProblem, MultiPointConstraint
import time

start_time = time.time()  # inizio misurazione


#-----------Import Mesh--------------------#
meshdata   = gmshio.read_from_msh("Foam1-16-R50-3.msh", MPI.COMM_WORLD, 0, gdim=3)
msh        = meshdata.mesh
cell_tags  = meshdata.cell_tags
facet_tags = meshdata.facet_tags


#----Definine Trial and Test function------#

V = functionspace(msh,("Lagrange",2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#---------------Dirichlet BC---------------#
wall_tag = 2  
wall_facets = facet_tags.find(wall_tag)
bc = dirichletbc(default_scalar_type(0), locate_dofs_topological(V, msh.topology.dim - 1, wall_facets), V)



#-----------Variational Problem--------------#    
f = Constant(msh,default_scalar_type(1))

a = ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx
L = f*v*ufl.dx


problem = LinearProblem(a,L,bcs=[bc],petsc_options_prefix="basic_linear_problem",petsc_options={"ksp_type":"preonly","pc_type":"lu","pc_factor_mat_solver_type": "mumps"})
uh = problem.solve()
uh.name = "u"


#-----------Assesment of Transport Parameters k0',alpha_th--------#
c = Constant(msh,default_scalar_type(1))
A = assemble_scalar(form(c*dx))
uh_int = assemble_scalar(form(uh*dx))
uh2_int = assemble_scalar(form(uh**2*dx))

phi      = 8*A/(2e-4)**3
k0p      = uh_int/A*phi
alpha_th = uh2_int/(uh_int**2)*A

print(phi)
print(alpha_th)
print(k0p)



#--------------Writing Scaled Thermal Field into a file----------#

with VTXWriter(msh.comm, "Solution Fields/Poisson.bp",[uh]) as vtx:
     vtx.write(0.0)


        
