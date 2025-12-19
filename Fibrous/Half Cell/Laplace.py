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
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx.common import Timer, list_timings

#-----------Import Mesh--------------------#

meshdata   = gmshio.read_from_msh("Fibrous-cell-half_meshbnd.msh", MPI.COMM_WORLD, 0, gdim=2)
msh        = meshdata.mesh
cell_tags  = meshdata.cell_tags
facet_tags = meshdata.facet_tags

#----Definine Trial and Test function------#

V = functionspace(msh,("Lagrange",2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

#---------------Dirichlet BC---------------#
bc = []

#--------------Periodicity-----------------#
tol = 250 * np.finfo(default_scalar_type).resolution

def periodic_boundary(x):
    on_xmax = np.isclose(x[0], 2e-4, atol=tol)
    not_on_ymin = np.logical_not(np.isclose(x[1], 0.0, atol=tol))
    not_on_ymax = np.logical_not(np.isclose(x[1], 2e-4, atol=tol))
    return np.logical_and(on_xmax, np.logical_and(not_on_ymin, not_on_ymax))

def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = 2e-4 - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x



with Timer("~PERIODIC: Initialize MPC"):
    mpc = MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation,bcs=None)
    mpc.finalize()

#-----------Variational Problem--------------# 
f = Constant(msh,default_scalar_type(0))
g = as_vector((1.0, 0.0))
ds = Measure("ds",domain=msh,subdomain_data=facet_tags)
n  = FacetNormal(msh)

a = ufl.dot(ufl.grad(u),ufl.grad(v))*ufl.dx
L = f*v*ufl.dx+ufl.dot(g,n)*v*(ds(6)+ds(7))

problem = LinearProblem(a,L,mpc,petsc_options={"ksp_type":"preonly","pc_type":"lu"})

uh = problem.solve()
uh.name = "u"

#-----------Assesment of Transport Parameters alpha_inf,Lv--------#
c = Constant(msh,default_scalar_type(1))

E = g-ufl.grad(uh)
A = assemble_scalar(form(c*dx))
S = assemble_scalar(form(c*(ds(6)+ds(7))))


E_mean = np.zeros(msh.geometry.dim)

for i in range(msh.geometry.dim):
    E_mean[i] = assemble_scalar(form(E[i] * dx)) / A


EE_mean = assemble_scalar(form(ufl.dot(E,E)*dx))/A
EE_mean2 = np.dot(E_mean, E_mean)

alpha_inf = EE_mean/(EE_mean2)

E_A = assemble_scalar(form(ufl.dot(E,E)*dx))
E_S = assemble_scalar(form(ufl.dot(E,E)*(ds(6)+ds(7))))
Lv = 2*E_A/E_S

print(alpha_inf)
print(Lv)

#--------------Writing Scaled Electric Field into a file----------#
with VTXWriter(msh.comm, "Solution Fields/Laplace.bp", [uh]) as vtx:
    vtx.write(0.0)
