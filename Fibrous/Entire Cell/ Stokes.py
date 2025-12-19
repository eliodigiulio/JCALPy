import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem import functionspace, Constant, dirichletbc, locate_dofs_topological, assemble_scalar,form, Function
import ufl
from ufl import dx, grad, inner, ds, Measure, FacetNormal
from basix.ufl import element, mixed_element
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx.common import Timer, list_timings
from dolfinx.mesh import locate_entities_boundary

#-----------Import Mesh--------------------#

meshdata   = gmshio.read_from_msh("Fibrous-cell_meshbnd.msh", MPI.COMM_WORLD, 0, gdim=2)
msh        = meshdata.mesh
cell_tags  = meshdata.cell_tags
facet_tags = meshdata.facet_tags

#----Definine Trial and Test function------#

Q_el = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P_el = element("Lagrange", msh.basix_cell(), 1)
V_el = mixed_element([Q_el, P_el])
V = functionspace(msh, V_el)


(u,p) = ufl.TrialFunctions(V)
(v,q) = ufl.TestFunctions(V)

V0 =   V.sub(0)
Q, _ = V.sub(0).collapse()

f_h1 = Function(Q)
f_h2 = Function(Q)

#---------------Dirichlet BC---------------#

bc2 = dirichletbc(f_h1, locate_dofs_topological((V0,Q), msh.topology.dim - 1, facet_tags.find(6)), V0)
bc3 = dirichletbc(f_h1, locate_dofs_topological((V0,Q), msh.topology.dim - 1, facet_tags.find(7)), V0)
bc = [bc2, bc3]


W0 =   V.sub(1)
K, _ = V.sub(1).collapse()

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

def periodic_boundary2(x):
    on_xmax = np.isclose(x[1], 2e-4, atol=tol)
    not_on_ymin = np.logical_not(np.isclose(x[0], 0.0, atol=tol))
    not_on_ymax = np.logical_not(np.isclose(x[0], 2e-4, atol=tol))
    return np.logical_and(on_xmax, np.logical_and(not_on_ymin, not_on_ymax))

def periodic_relation2(x):
    out_x = np.zeros_like(x)
    out_x[0] = x[0]
    out_x[1] = 2e-4 - x[1]
    out_x[2] = x[2]
    return out_x


with Timer("~PERIODIC: Initialize MPC"):
    mpc = MultiPointConstraint(V)
    mpc.create_periodic_constraint_geometrical(V0.sub(0), periodic_boundary, periodic_relation,bcs=bc)
    mpc.create_periodic_constraint_geometrical(V0.sub(0), periodic_boundary2, periodic_relation2,bcs=bc)
    mpc.create_periodic_constraint_geometrical(V0.sub(1), periodic_boundary, periodic_relation,bcs=bc)
    mpc.create_periodic_constraint_geometrical(V0.sub(1), periodic_boundary2, periodic_relation2,bcs=bc)
    mpc.create_periodic_constraint_geometrical(W0, periodic_boundary, periodic_relation,bcs=bc)
    mpc.create_periodic_constraint_geometrical(W0, periodic_boundary2, periodic_relation2,bcs=bc)
    mpc.finalize()




#-----------Variational Problem--------------# 

f  = Constant(msh,(default_scalar_type(1),default_scalar_type(0)))
mu = Constant(msh,default_scalar_type(1))

a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx 
     + ufl.div(v) * p * ufl.dx 
     - q * ufl.div(u) * ufl.dx)

L = ufl.inner(f, v) * ufl.dx


problem = LinearProblem(a, L, mpc, bcs=bc, petsc_options={"ksp_type": "preonly", "pc_type": "lu","pc_factor_mat_solver_type": "mumps"})                                                   
uh = problem.solve()
uh_u,uh_p = uh.sub(0).collapse(), uh.sub(1).collapse()

uh_u.name = "u"
uh_p.name = "p"

#-----------Assesment of Transport Parameters alpha_v,k0--------#
c = Constant(msh,default_scalar_type(1))
A = assemble_scalar(form(c*dx))

k0_mean = np.zeros(msh.geometry.dim)

for i in range(msh.geometry.dim):
    k0_mean[i] = assemble_scalar(form(uh_u[i] * dx)) / A

EE_mean = assemble_scalar(form(ufl.dot(uh_u,uh_u)*dx))/A
EE_mean2 = np.dot(k0_mean, k0_mean)

alpha_v = EE_mean/(EE_mean2)

e = ufl.as_vector((1.0, 0.0))
phi = 0.9195
k0 = assemble_scalar(form(ufl.dot(uh_u,e)*dx))/A*phi

print(alpha_v)
print(k0)


with VTXWriter(msh.comm, "Solution Fields/Stokes-v.bp",[uh_u]) as vtx:
     vtx.write(0.0)
with VTXWriter(msh.comm, "Solution Fields/Stokes-p.bp",[uh_p]) as vtx:
     vtx.write(0.0)
