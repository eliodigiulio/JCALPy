
import gmsh
import sys
import os

gmsh.initialize(sys.argv)
gmsh.option.setString("Geometry.OCCTargetUnit", "M")

path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, "Foam1_16-R50-3.step"))

Lc = 2e-4
gmsh.option.setNumber("Mesh.MeshSizeMax", Lc/100)

gmsh.model.addPhysicalGroup(3, [1], 1, "Domain")
gmsh.model.addPhysicalGroup(2, [7, 16, 17, 18], 2, "Wall")
gmsh.model.addPhysicalGroup(2, [3, 11, 12], 3, "LateralXZ")
gmsh.model.addPhysicalGroup(2, [1, 6 ,8], 4, "LateralYZ")
gmsh.model.addPhysicalGroup(2, [10, 19], 5, "Inlet")
gmsh.model.addPhysicalGroup(2, [5, 15], 6, "SideXZ")
gmsh.model.addPhysicalGroup(2, [2, 9], 7, "Up")
gmsh.model.addPhysicalGroup(2, [4, 13, 14], 8, "Down")


# 🔹 Generate mesh
gmsh.model.mesh.generate(3)
gmsh.write(os.path.join(path, "Foam1-16-R50-3.msh"))

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