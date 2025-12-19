import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem import functionspace, Constant,Function, dirichletbc, locate_dofs_topological, assemble_scalar,form, Expression
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import dx, grad, inner, ds, Measure, FacetNormal, as_vector
#from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx.common import Timer, list_timings

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
wall_tag = [5,4]  
wall_facets = np.hstack([facet_tags.find(tag) for tag in wall_tag])
bc1 = dirichletbc(default_scalar_type(0), locate_dofs_topological(V, msh.topology.dim - 1, facet_tags.find(5)), V)
bc2 = dirichletbc(default_scalar_type(0), locate_dofs_topological(V, msh.topology.dim - 1, facet_tags.find(4)), V)
bc = [bc1,bc2]

#-----------Variational Problem--------------# 
f = Constant(msh,default_scalar_type(0))
g = as_vector((1.0, 0.0,0.0))
ds = Measure("ds",domain=msh,subdomain_data=facet_tags)
n  = FacetNormal(msh)


a = ufl.dot(ufl.grad(u),ufl.grad(v))*ufl.dx
L = f*v*ufl.dx+ufl.dot(g,n)*v*(ds(2))

problem = LinearProblem(a,L,bcs=bc,petsc_options_prefix="basic_linear_problem",petsc_options={"ksp_type":"preonly","pc_type":"lu","pc_factor_mat_solver_type": "mumps"})

uh = problem.solve()
uh.name = "u"

#-----------Assesment of Transport Parameters alpha_inf,Lv--------#
c = Constant(msh,default_scalar_type(1))

E = g-ufl.grad(uh)
A = assemble_scalar(form(c*dx))
S = assemble_scalar(form(c*(ds(2))))


E_mean = np.zeros(msh.geometry.dim)

for i in range(msh.geometry.dim):
    E_mean[i] = assemble_scalar(form(E[i] * dx)) / A

e  = ufl.as_vector((1.0, 0.0, 0.0))
EE_mean = assemble_scalar(form(ufl.dot(E,E)*dx))/A
EE_mean2 = np.dot(E_mean, e)**2

alpha_inf = EE_mean/(EE_mean2)

E_A = assemble_scalar(form(ufl.dot(E,E)*dx))
E_S = assemble_scalar(form(ufl.dot(E,E)*(ds(2))))
Lv = 2*E_A/E_S

print(alpha_inf)
print(Lv)

#--------------Writing Scaled Electric Field into a file----------#
with VTXWriter(msh.comm, "Solution Fields/Laplace.bp", [uh]) as vtx:
    vtx.write(0.0)
