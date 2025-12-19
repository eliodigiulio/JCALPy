import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import default_scalar_type, common
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.io import gmsh as gmshio
from dolfinx.fem import functionspace, Constant, dirichletbc, locate_dofs_topological,assemble_scalar,form, Function
import ufl
from ufl import dx, grad, inner, div, ds, Measure, FacetNormal, Identity, outer, extract_blocks, MixedFunctionSpace
from basix.ufl import element, mixed_element
import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem, MultiPointConstraint
from dolfinx.common import Timer, list_timings
from dolfinx.mesh import locate_entities_boundary
#from dolfinx.fem.petsc import LinearProblem
from ufl.core.expr import Expr


#-----------Import Mesh--------------------#

meshdata   = gmshio.read_from_msh("Foam1-8-R50-3.msh", MPI.COMM_WORLD, 0, gdim=3)
msh        = meshdata.mesh
cell_tags  = meshdata.cell_tags
facet_tags = meshdata.facet_tags

#----Definine Trial and Test function------#

Q_el = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P_el = element("Lagrange", msh.basix_cell(), 1)

U = functionspace(msh, Q_el)
P = functionspace(msh, P_el)
V = MixedFunctionSpace(U, P)

(u,p) = ufl.TrialFunctions(V)
(v,q) = ufl.TestFunctions(V)


#---------------Dirichlet BC---------------#
# Wall
f_h1 = Function(U)
wall = locate_dofs_topological(U, 2, facet_tags.find(2))
bc1  = dirichletbc(f_h1,wall)


#---------------Antisymmetry BC---------------#
Uy   = U.sub(1)
Qy,_ = U.sub(1).collapse()
f_h2 = Function(Qy)

Uz   = U.sub(2)
Qz,_ = U.sub(2).collapse()
f_h3 = Function(Qy)

antisymmetry5y = locate_dofs_topological((Uy,Qy), 2, facet_tags.find(5))
antisymmetry6y = locate_dofs_topological((Uy,Qy), 2, facet_tags.find(6))
antisymmetry5z = locate_dofs_topological((Uy,Qy), 2, facet_tags.find(5))
antisymmetry6z = locate_dofs_topological((Uy,Qy), 2, facet_tags.find(6))
bc5y = dirichletbc(f_h2,antisymmetry5y,Uy)
bc6y = dirichletbc(f_h2,antisymmetry6y,Uy)
bc5z = dirichletbc(f_h3,antisymmetry5z,Uz)
bc6z = dirichletbc(f_h3,antisymmetry6z,Uz)
bcs = [bc1, bc5y, bc6y, bc5z, bc6z]

#---------------Symmetry BC---------------#
symmetry_marker3 = 3
symmetry_marker4 = 4
symmetry_marker7 = 7
symmetry_marker8 = 8
symmetry_marker5 = 5
symmetry_marker6 = 6
n3 = dolfinx_mpc.utils.create_normal_approximation(U, facet_tags, symmetry_marker3)
n4 = dolfinx_mpc.utils.create_normal_approximation(U, facet_tags, symmetry_marker4)
n7 = dolfinx_mpc.utils.create_normal_approximation(U, facet_tags, symmetry_marker7)
n8 = dolfinx_mpc.utils.create_normal_approximation(U, facet_tags, symmetry_marker8)


with common.Timer("~Stokes: Create slip constraint"):
    mpc_u = MultiPointConstraint(U)
    mpc_u.create_slip_constraint(U, (facet_tags, symmetry_marker3), n3, bcs=bcs)
    mpc_u.create_slip_constraint(U, (facet_tags, symmetry_marker4), n4, bcs=bcs)
    mpc_u.create_slip_constraint(U, (facet_tags, symmetry_marker7), n7, bcs=bcs)
    mpc_u.create_slip_constraint(U, (facet_tags, symmetry_marker8), n8, bcs=bcs)

mpc_u.finalize()

mpc_p = MultiPointConstraint(P)
mpc_p.finalize()



def tangential_proj(u: Expr, n: Expr):
    return (Identity(u.ufl_shape[0]) - outer(n, n)) * u

def epsilon(u: Expr):
    return ufl.sym(grad(u))

def sigma(u: Expr, p: Expr, mu: Expr):
    return 2 * mu * epsilon(u) - p * Identity(u.ufl_shape[0])

#-----------Variational Problem--------------# 

f  = Constant(msh,(default_scalar_type(1),default_scalar_type(0),default_scalar_type(0)))
mu = Constant(msh,default_scalar_type(1))



a = (2* mu * inner(epsilon(u), epsilon(v)) - inner(p, ufl.div(v)) - ufl.inner(ufl.div(u), q)) * dx
L = inner(f, v) * dx + inner(Constant(msh, default_scalar_type(0.0)), q) * dx


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

c  = Constant(msh,default_scalar_type(1))
dx = dx(metadata={"quadrature_degree":7})
A  = assemble_scalar(form(c*dx))
e  = ufl.as_vector((1.0, 0.0, 0.0))

k0_mean = np.zeros(msh.geometry.dim)

for i in range(msh.geometry.dim):
    k0_mean[i] = assemble_scalar(form(uh_u[i] * dx)) / A

EE_mean = assemble_scalar(form(ufl.dot(uh_u,uh_u)*dx))/A
EE_mean2 = np.dot(k0_mean, k0_mean)


phi = 4*A/(2e-4)**3
alpha_v = EE_mean/(EE_mean2)
k0  = assemble_scalar(form(ufl.dot(uh_u,e)*dx))/A*phi

print(phi)
print(alpha_v)
print(k0)

#-----------Saving p and u fields for visualization--------#

with VTXWriter(msh.comm, "Solution Fields/Stokes-v.bp",[uh_u]) as vtx:
     vtx.write(0.0)
with VTXWriter(msh.comm, "Solution Fields/Stokes-p.bp",[uh_p]) as vtx:
     vtx.write(0.0)
