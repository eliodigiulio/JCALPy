import gmsh
import sys
import os

gmsh.initialize(sys.argv)
gmsh.option.setString("Geometry.OCCTargetUnit", "M")

path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, "Foam1_8-R50.step"))

Lc = 2e-4
gmsh.option.setNumber("Mesh.MeshSizeMax", Lc/50)

gmsh.model.addPhysicalGroup(3, [1], 1, "Domain")
gmsh.model.addPhysicalGroup(2, [1,5,9,10,11,21,22,23], 2, "Wall")
gmsh.model.addPhysicalGroup(2, [4,8,17,18,20], 3, "SymmetryDown")
gmsh.model.addPhysicalGroup(2, [2,6,12,14,15], 4, "SymmetryLateral")
gmsh.model.addPhysicalGroup(2, [3,7], 5, "Inlet")
gmsh.model.addPhysicalGroup(2, [16,27], 6, "Outlet")
gmsh.model.addPhysicalGroup(2, [13,25,26], 7, "Up")
gmsh.model.addPhysicalGroup(2, [19,24,28], 8, "Side")


# 🔹 Generate mesh
gmsh.model.mesh.generate(3)
gmsh.write(os.path.join(path, "Foam1-8-R50-3.msh"))

gmsh.fltk.run()


# Nodes
nodes = gmsh.model.mesh.getNodes()
print("Numero di nodi:", len(nodes[0]))

# Elementi per dimensione (0 = punti, 1 = linee, 2 = superfici, 3 = volume)
for dim in [1, 2, 3]:
    types, elementTags, _ = gmsh.model.mesh.getElements(dim)
    num_elements = sum(len(tags) for tags in elementTags)
    print(f"Numero elementi {dim}D:", num_elements)

gmsh.finalize()


