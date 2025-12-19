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
from dolfinx_mpc import LinearProblem, MultiPointConstraint


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
wall_ids    = [6,7]
wall_facets = np.hstack([facet_tags.find(i) for i in wall_ids])
bc = dirichletbc(default_scalar_type(0), locate_dofs_topological(V, msh.topology.dim - 1, wall_facets), V)

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
    mpc.create_periodic_constraint_geometrical(V, periodic_boundary, periodic_relation, bcs=[bc])
    mpc.finalize()


#-----------Variational Problem--------------# 
f = Constant(msh,default_scalar_type(1))

a = ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx
L = f*v*ufl.dx


problem = LinearProblem(a,L,mpc,bcs=[bc],petsc_options={"ksp_type":"preonly","pc_type":"lu"})
uh      = problem.solve()
uh.name = "u"

#-----------Assesment of Transport Parameters k0',alpha_th--------#
c = Constant(msh,default_scalar_type(1))
A = assemble_scalar(form(c*dx))
uh_int = assemble_scalar(form(uh*dx))
uh2_int = assemble_scalar(form(uh**2*dx))

k0p = uh_int/A*0.9195
alpha_th = uh2_int/(uh_int**2)*A

print(alpha_th)
print(k0p)

#--------------Writing Scaled Thermal Field into a file----------#
with VTXWriter(msh.comm, "Solution Fields/Poisson.bp",[uh]) as vtx:
     vtx.write(0.0)


        