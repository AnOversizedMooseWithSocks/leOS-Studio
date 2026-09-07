import numpy as np
class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, float).copy()
        self.faces = [tuple(int(i) for i in f) for f in faces]
    @property
    def n_faces(self): return len(self.faces)
    @property
    def n_vertices(self): return len(self.vertices)
    def copy(self): return Mesh(self.vertices.copy(), list(self.faces))
    def triangulate(self):
        """Fan-triangulate every face; returns a list of (a,b,c) index tuples."""
        out = []
        for f in self.faces:
            for k in range(1, len(f) - 1):
                out.append((f[0], f[k], f[k + 1]))
        return out
def box(x=1.0, y=None, z=None):
    """hm.box(x, y, z) -- FULL extents, matching the engine."""
    y = x if y is None else y; z = x if z is None else z
    a, b, c = float(x)/2, float(y)/2, float(z)/2
    V = np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                  [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]], float)
    F = [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]
    return Mesh(V, F)
cube = box

def tetrahedron(r=1.0):
    r = float(r)
    V = np.array([[r,r,r],[r,-r,-r],[-r,r,-r],[-r,-r,r]], float)
    return Mesh(V, [(0,1,2),(0,3,1),(0,2,3),(1,3,2)])

def grid(nx=6, ny=6, w=1.0, h=1.0):
    nx, ny = int(nx), int(ny)
    xs = np.linspace(-w/2, w/2, nx+1); zs = np.linspace(-h/2, h/2, ny+1)
    V = np.array([[x, 0.0, z] for z in zs for x in xs], float)
    F = [(j*(nx+1)+i, j*(nx+1)+i+1, (j+1)*(nx+1)+i+1, (j+1)*(nx+1)+i)
         for j in range(ny) for i in range(nx)]
    return Mesh(V, F)
