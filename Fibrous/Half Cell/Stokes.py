import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import default_scalar_type
from dolfinx.io import gmsh as gmshio
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem import functionspace, Constant, dirichletbc, locate_dofs_topological, assemble_scalar,form, Function, locate_dofs_geometrical
import ufl
import dolfinx_mpc.utils
from ufl import dx, grad, inner, ds, Measure, FacetNormal, extract_blocks, MixedFunctionSpace
from basix.ufl import element, mixed_element
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx.common import Timer, list_timings
from dolfinx.mesh import locate_entities_boundary

#-----------Import Mesh--------------------#

meshdata   = gmshio.read_from_msh("Fibrous-cell-half_meshbnd.msh", MPI.COMM_WORLD, 0, gdim=2)
msh        = meshdata.mesh
cell_tags  = meshdata.cell_tags
facet_tags = meshdata.facet_tags
# vertex_tags = meshdata.vertex_tags

#----Definine Trial and Test function------#

Q_el = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P_el = element("Lagrange", msh.basix_cell(), 1)

U = functionspace(msh, Q_el)
P = functionspace(msh, P_el)
V = MixedFunctionSpace(U, P)

(u,p) = ufl.TrialFunctions(V)
(v,q) = ufl.TestFunctions(V)



#---------------Dirichlet BC---------------#

f_h1 = Function(U)

bc2 = dirichletbc(f_h1, locate_dofs_topological(U, 1, facet_tags.find(6)))
bc3 = dirichletbc(f_h1, locate_dofs_topological(U, 1, facet_tags.find(7)))
bcs = [bc2, bc3]




#--------------Periodicity-----------------#
tol = 250 * np.finfo(default_scalar_type).resolution

def periodic_boundary(x):
    on_xmax = np.isclose(x[0], 2e-4, atol=tol)
    not_on_ymin = np.logical_not(np.isclose(x[1], 0.0, atol=tol))
    not_on_ymax = np.logical_not(np.isclose(x[1], 2e-4, atol=tol))
    return np.logical_and(on_xmax, np.logical_and(not_on_ymin, not_on_ymax))
# def periodic_boundary(x):
#     return np.isclose(x[0], 2e-4, atol=tol)

def periodic_relation(x):
    out_x = np.zeros_like(x)
    out_x[0] = 2e-4 - x[0]
    out_x[1] = x[1]
    out_x[2] = x[2]
    return out_x

symmetry_marker4 = 4
symmetry_marker5 = 5
n4 = dolfinx_mpc.utils.create_normal_approximation(U, facet_tags, symmetry_marker4)
n5 = dolfinx_mpc.utils.create_normal_approximation(U, facet_tags, symmetry_marker5)


with Timer("~PERIODIC: Initialize MPC"):
    mpc_u = MultiPointConstraint(U)
    mpc_u.create_periodic_constraint_geometrical(U, periodic_boundary, periodic_relation,bcs=bcs)
    mpc_u.create_slip_constraint(U, (facet_tags, symmetry_marker4), n4, bcs=bcs)
    mpc_u.create_slip_constraint(U, (facet_tags, symmetry_marker5), n5, bcs=bcs)
mpc_u.finalize()


mpc_p = MultiPointConstraint(P)
# mpc_p.create_periodic_constraint_geometrical(P, periodic_boundary, periodic_relation,bcs=[])
mpc_p.finalize()


#-----------Variational Problem--------------# 

f  = Constant(msh,(default_scalar_type(1),default_scalar_type(0)))
mu = Constant(msh,default_scalar_type(1))

a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx 
     + ufl.div(v) * p * ufl.dx 
     - q * ufl.div(u) * ufl.dx)

L = ufl.inner(f, v) * ufl.dx+ inner(Constant(msh, default_scalar_type(0.0)), q) * dx


#------------------Solver--------------------# 

tol = np.finfo(msh.geometry.x.dtype).eps * 1000
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
    "ksp_monitor": None,
}
problem = LinearProblem(extract_blocks(a), extract_blocks(L), [mpc_u, mpc_p], bcs=bcs, petsc_options=petsc_options)

uh_u, uh_p = problem.solve()
uh_u.name  = "u"
uh_p.name  = "p"


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


with VTXWriter(msh.comm, "Solution Fields/Stokes-v2.bp",[uh_u]) as vtx:
     vtx.write(0.0)
with VTXWriter(msh.comm, "Solution Fields/Stokes-p2.bp",[uh_p]) as vtx:
     vtx.write(0.0)
