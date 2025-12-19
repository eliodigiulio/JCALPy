import gmsh
import sys
import os

gmsh.initialize(sys.argv)
gmsh.model.add("Fibrous-cell")
gmsh.option.setString("Geometry.OCCTargetUnit", "M")

path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, "Fibrous-cell.step"))

gmsh.option.setNumber("Mesh.MeshSizeMax", 0.000001)

gmsh.model.addPhysicalGroup(2, [1], 1, "Domain")
gmsh.model.addPhysicalGroup(1, [2], 2, "Inlet")
gmsh.model.addPhysicalGroup(1, [4], 3, "Outlet")
gmsh.model.addPhysicalGroup(1, [1], 4, "Wall_up")
gmsh.model.addPhysicalGroup(1, [3], 5, "Wall_down")
gmsh.model.addPhysicalGroup(1, [5], 6, "Wall_cyl1")
gmsh.model.addPhysicalGroup(1, [6], 7, "Wall_cyl2")

# 🔹 Recupera le curve dei due cilindri
cyl1 = gmsh.model.getEntitiesForPhysicalGroup(1, 6)
cyl2 = gmsh.model.getEntitiesForPhysicalGroup(1, 7)

cyl3 = gmsh.model.getEntitiesForPhysicalGroup(1, 5)
curves = cyl1 + cyl2


# Cyl1
bl1 = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(bl1, "CurvesList", cyl1)
gmsh.model.mesh.field.setNumber(bl1, "hwall_n", 1e-7)
gmsh.model.mesh.field.setNumber(bl1, "thickness", 5e-6)
gmsh.model.mesh.field.setNumber(bl1, "ratio", 1.3)
gmsh.model.mesh.field.setAsBoundaryLayer(bl1)

# Cyl2
bl2 = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(bl2, "CurvesList", cyl2)
gmsh.model.mesh.field.setNumber(bl2, "hwall_n",1e-7)
gmsh.model.mesh.field.setNumber(bl2, "thickness", 5e-6)
gmsh.model.mesh.field.setNumber(bl2, "ratio", 1.3)
gmsh.model.mesh.field.setAsBoundaryLayer(bl2)


# Generate mesh
gmsh.model.mesh.generate(2)
gmsh.write(os.path.join(path, "Mesh/Fibrous-cell_meshbnd.msh"))

gmsh.fltk.run()
gmsh.finalize()

