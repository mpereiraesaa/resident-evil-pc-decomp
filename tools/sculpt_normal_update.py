"""Carry smooth viewport normals through an analytic mesh deformation.

RE1's OBJ export has disconnected triangle corners but already-smoothed TMD
normals. Recalculating vertex normals in Blender makes those corners faceted.
For a smooth displacement F(p), transform each original normal by the inverse
transpose of F's local Jacobian instead.
"""

from mathutils import Matrix, Vector


def displaced_normal(position, normal, displace, step=0.25):
    point = Vector(position)
    old = Vector(normal)
    if old.length_squared < 1e-12:
        return old
    partials = []
    for axis in range(3):
        delta = Vector((0.0, 0.0, 0.0))
        delta[axis] = step
        partials.append((Vector(displace(point + delta)) -
                         Vector(displace(point - delta))) / (2.0 * step))
    jacobian = Matrix(((1.0 + partials[0].x, partials[1].x, partials[2].x),
                       (partials[0].y, 1.0 + partials[1].y, partials[2].y),
                       (partials[0].z, partials[1].z, 1.0 + partials[2].z)))
    if abs(jacobian.determinant()) < 1e-6:
        raise ValueError("sculpt deformation folds the mesh")
    return (jacobian.inverted().transposed() @ old).normalized()
