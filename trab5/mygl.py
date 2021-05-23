import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
TOL = sys.float_info.epsilon

from algebra import unitary, vector, dot, reflect, to_projective, to_cartesian, transf4x3, cross, change_basis_matrix, translation_matrix, identity_matrix


class Render:
    def __init__(self,scene,bck_color):
        self.scene = scene
        camera = scene.get_camera()
        h = camera.get_h()
        w = camera.get_w()
        self.frame_buffer = np.full((h,w,3),bck_color,dtype=np.float64)
        self.z_buffer = np.full((h,w),np.inf,dtype=np.float64)
        self.view_matrix = camera.get_look_at()
        self.projection_matrix = camera.get_projection()
    
    
    def phong_shade(self,vertex,normal,material,lights):
        ka, kd, ks, ns = material.get_values()
        ve = unitary(-vertex)
        color = vector(0, 0, 0)
        
        for light in lights:
            pl, la, li = light.get_values()
            color += la * ka
            vl = unitary(pl - vertex)
            cos_diff = dot(vl, normal)
            
            if cos_diff > 0:
                color += li * kd * cos_diff
                rl = reflect(vl, normal)
                cos_spec = dot(rl, ve)
                
                if cos_spec > 0:
                    color += li * ks * (cos_spec ** ns)
                    
        return color
    
    
    def getABC(self, v0, v1, v2):
        y0 = v0[1]
        y1 = v1[1]
        y2 = v2[1]
        
        if y0 > y1:
            if y1 > y2:
                return v0, v1, v2
            elif y0 > y2:
                return v0, v2, v1
            else:
                return v2, v0, 1
        else:
            if y0 > y2:
                return v1, v0, v2
            elif y1 > y2:
                return v1, v2, v0
            else:
                return v2, v1, v0


    def raster_triang(self,v0,v1,v2):  # vi = (x,y,1/z,r,g,b)
        vA, vB, vC = self.getABC(v0, v1, v2)
        yA, yB, yC = int(vA[1]), int(vB[1]), int(vC[1])
        
        a = np.linspace(vC, vB, yB - yC, endpoint = False)
        b = np.linspace(vC, vA, yA - yC, endpoint = False)
        c = np.linspace(vB, vA, yA - yB, endpoint = False)
        ac = np.concatenate((a, c), axis = 0)
        
        if (vB[0] - vA[0]) * (vC[1] - vA[1]) - (vC[0] - vA[0]) * (vB[1] - vA[1]) > 0:
            left = ac
            right = b
        else:
            left = b
            right = ac
        
        #left[:, 0] = np.rint(left[:, 0])
        #right[:, 0] = np.rint(right[:, 0])
        
        fragments = []
        
        for i in range(yA - yC):
            fragments_hline = np.linspace(left[i], right[i], int(round(right[i, 0]) - round(left[i, 0])), endpoint = False)
            fragments_hline[:, 2] = 1.0 / fragments_hline[:, 2]
            fragments.append(fragments_hline)
            
        fragments = np.array(fragments)
        
        return fragments
    

    def render_zb(self):
        objects = self.scene.get_objects()
        lights = self.scene.get_lights()
        camera = self.scene.get_camera()
        
        for object in objects:
            vertices_scene = object.get_vertices()
            normals_scene = object.get_normals()
            material = object.get_material()
            
            # to camera coordinate system
            vertices_eye = []
            
            for vertex in vertices_scene:
                vertices_eye.append(transf4x3(self.view_matrix, vertex))
                
            vertices_eye = np.array(vertices_eye)
            normals_eye = []
            MiT = np.linalg.inv(self.view_matrix.T)
            
            for normal in normals_scene:
                n = transf4x3(MiT, normal)
                normals_eye.append(unitary(n))
                
            normals_eye = np.array(normals_eye)
            
            # Vertex colors
            vertex_colors = []
            
            for i, vertex in enumerate(vertices_eye):
                normal = normals_eye[i]
                color = self.phong_shade(vertex, normal, material, lights)
                vertex_colors.append(color)
                
            vertex_colors = np.array(vertex_colors)
            
            # Projection
            vertices_proj = []
            
            for vertex in vertices_eye:
                vp = to_projective(vertex)
                vproj = np.dot(self.projection_matrix, vp)
                vertices_proj.append(vproj)
                
            vertices_proj = np.array(vertices_proj)
            
            # Clip
            
            
            # Canvas
            vertices = []
            
            for i, vertex in enumerate(vertices_proj):
                vn = to_cartesian(vertex)
                vc = camera.to_canvas(vn)
                vtx = np.array([vc[0], vc[1], 1 / vc[2], vertex_colors[0], vertex_colors[1], vertex_colors[2]])
                vertices.append(vtx)
            vertices = np.array(vertices)
            
            # Fragments from the primitive (triangle)
            triangles = object.get_triangles()
            
            for t in triangles:
                v0 = vertices[t[0]]
                v1 = vertices[t[1]]
                v2 = vertices[t[2]]
                fragments = self.raster_triang(v0, v1, v2)
            
            for fragment in fragments:
                ix = int(fragment[0])
                iy = int(fragment[1])
                z = fragment[2]
                color = fragment[3:]
                
                if z < self.z_buffer[iy, ix]:
                    self.frame_buffer[iy, ix] = color
                    self.z_buffer[iy, ix] = z
            
        return
            
    
    def get_frame_buffer(self):
        return self.frame_buffer
    
    
    def get_z_buffer(self):
        return self.z_buffer



class Camera:
    def __init__(self,fov,w,h,near,far,eye,at,up):
        self.fov = fov
        self.w = w
        self.h = h
        self.near = near
        self.far = far
        self.a = 2*near*np.tan(fov*np.pi/360)
        self.b = self.a*w/h
        self.eye = eye
        self.ze = unitary(at-eye)
        self.xe = unitary(cross(self.ze,up))
        self.ye = cross(self.ze,self.xe)
    
    
    def show_values(self):
        print("CAMERA:")
        print(f'fov={self.fov},  n={self.near}, f={self.far}')
        print(f'(w,h)=({self.w},{self.h})')
        print(f'(b,a)=({self.b},{self.a})')
        print(f'xe={self.xe}')
        print(f'ye={self.ye}')
        print(f'ze={self.ze}')
        
        
    def get_look_at(self):
        R = change_basis_matrix(self.xe,self.ye,self.ze)
        T = translation_matrix(-self.eye[0],-self.eye[1],-self.eye[2])
        return np.dot(R,T)
    
    
    def get_frustrum(self):
        left = -self.b/2
        right = +self.b/2
        top = -self.a/2
        bottom = self.a/2
        P = identity_matrix()
        P[0,0]=2*self.near/(right-left)
        P[0,2]=-(right+left)/(right-left)
        P[1,1]=2*self.near/(bottom-top)
        P[1,2]=-(bottom+top)/(bottom-top)
        P[2,2]=(self.far+self.near)/(self.far -self.near)
        P[2,3]=-2*self.far*self.near/(self.far -self.near)
        P[3,2]=1
        P[3,3]=0
        return P
         
    
    def get_projection(self):
        P = identity_matrix()
        aspecto = self.w/self.h
        P[0,0]=1/(aspecto*np.tan(self.fov*np.pi/360))
        P[1,1]=1/np.tan(self.fov*np.pi/360)
        P[2,2]=(self.far+self.near)/(self.far -self.near)
        P[2,3]=-2*self.far*self.near/(self.far -self.near)
        P[3,2]=1
        P[3,3]=0
        return P
    
    
    def to_canvas(self,vn):
        xc = round((self.w-1)*(vn[0]+1.0)/2.0)
        yc = round((self.h-1)*(vn[1]+1.0)/2.0)
        zc = (vn[2]+1.0)/2.0
        return np.array([xc,yc,zc],dtype=np.float64)
        
                              
    def ray_to(self,x,y):
        dx = self.b*(x/self.w-0.5)
        dy = self.a*(y/self.h-0.5)
        dz = self.near
        ray = dx*self.xe+dy*self.ye+dz*self.ze
        return ray
    
    
    def get_eye(self):
        return self.eye
    
    
    def get_w(self):
        return self.w
    
    
    def get_h(self):
        return self.h
    
    
    
class ObjectBuffer:
    def __init__(self, vertices, triangles, normals, material,model_matrix=identity_matrix()):
        self.vertices = copy.deepcopy(vertices)
        self.model_matrix = model_matrix
        self.triangles = copy.deepcopy(triangles)
        self.normals = copy.deepcopy(normals)
        self.material = material
    
    
    def show_values(self):
        print("Vertices and normals: ")
        for i in range(self.vertices.shape[0]):
            print(f"\t v{i} = {self.vertices[i,:]} n{i} = {self.normals[i, :]}")

        print("Triangles: ")
        for i in range(self.triangles.shape[0]):
            print(f"\t t{i} = {self.triangles[i,:]}")
            
            
    def set_model_matrix(self,matrix):
        self.model_matrix = matrix
    
    
    def get_vertices(self):
        vertices = []
        for vertex in self.vertices:
            vertices.append(transf4x3(self.model_matrix,vertex))
        
        return np.array(vertices)
    
    
    def get_normals(self):
        normals = []
        MiT = np.linalg.inv(self.model_matrix.T)
        for normal in self.normals:
            n = transf4x3(MiT,normal)
            normals.append(unitary(n))
        return np.array(normals)


    def get_triangles(self):
        return self.triangles


    def get_material(self):
        return self.material
        
    
    
class UnityCube(ObjectBuffer):
    def __init__(self,material):
        vertices = np.array([
            # Front face
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5], 
            # Front face
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
            # Top face
            [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
            # Bottom face
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
            # Right face
            [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5],
            # Left face
            [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5]
            ],dtype=np.float64)
        normals = np.array([
            #Front
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            #Back
            [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0],
            #Top
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            #Bottom
            [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0], [0.0, -1.0, 0.0],
            #Right
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            #Left
            [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]
            ], dtype=np.float64)
        triangles = np.array([
            # front
            [0,  1,  2], [0,  2,  3],
            # back
            [4,  5,  6], [4,  6,  7],
            # top
            [8,  9,  10], [8,  10, 11],
            # bottom
            [12, 13, 14], [12, 14, 15],
            # right
            [16, 17, 18], [16, 18, 19],
            # left
            [20, 21, 22], [20, 22, 23]
            ],dtype=np.int32)
        super().__init__(vertices,triangles,normals,material)



class Icosahedron(ObjectBuffer):
    def __init__(self,material):
        X=0.525731112119133606 
        Z=0.850650808352039932
        pts = np.array([
            [-X, 0.0, Z], [X, 0.0, Z], [-X, 0.0, -Z], [X, 0.0, -Z],
            [0.0, Z, X], [0.0, Z, -X], [0.0, -Z, X], [0.0, -Z, -X],
            [Z, X, 0.0], [-Z, X, 0.0], [Z, -X, 0.0], [-Z, -X, 0.0] 
        ],dtype=np.float64)
        faces = np.array([
            [0,4,1], [0,9,4], [9,5,4], [4,5,8], [4,8,1], 
            [8,10,1], [8,3,10], [5,3,8], [5,2,3], [2,7,3],
            [7,10,3], [7,6,10], [7,11,6], [11,0,6], [0,1,6],
            [6,1,10], [9,0,11], [9,11,2], [9,2,5], [7,2,11] 
        ],dtype=np.int32)
        super().__init__(pts,faces,pts,material)



class Material:
    def __init__(self,ambient,diffuse,specular,shinesss):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shiness = shinesss


    def show_values(self):
        print("Material")
        print(f'ambient={self.ambient}')
        print(f'diffuse={self.diffuse}')
        print(f'specular={self.specular}')
        print(f'shinesss={self.shiness}')


    def get_values(self):
        return self.ambient,self.diffuse,self.specular,self.shiness



class Light:
    def __init__(self,position,ambient,intensity):
        self.position = position
        self.ambient = ambient
        self.intensity = intensity


    def show_values(self):
        print(f'position={self.position}')
        print(f'ambient={self.ambient}')
        print(f'intensity={self.intensity}')


    def get_values(self):
        return self.position,self.ambient,self.intensity



class Scene:
    def __init__(self,camera,objects,materials,lights):
        self.camera = camera
        self.objects = objects  # list of objects
        self.materials = materials # list of materials
        self.lights = lights # list of lights


    def show_values(self):
        self.camera.show_values()
        for object in self.objects:
            self.object.show_values()
        for material in self.materials:
            self.material.show_values()
        for light in self.lights:
            self.light.show_values()


    def get_camera(self):
        return self.camera


    def get_objects(self):
        return self.objects


    def get_materials(self):
        return self.materials


    def get_lights(self):
        return self.lights



def main():
    print("Hello World!")
    brass=Material(vector(0.33, 0.22, 0.03),vector(0.78, 0.57, 0.11),vector(1.0, 1.0, 1.0), 27.8)
    camera=Camera(45,640,480,0.5,10,vector(1.5,1.5,1.2), vector(0,0,0), vector(0,0,1))
    cube = UnityCube(brass)
    light = Light(vector(0.5,-2.,0.),vector(0.2,0.2,0.2),vector(1,1,1))
    scene = Scene(camera,[cube],[brass],[light])
    render = Render(scene,vector(0.4,0.4,0.4))
    render.render_zb()
    fbuffer = render.get_frame_buffer()
    plt.imshow(fbuffer)
    plt.imsave("fbuffer.png",fbuffer)
    plt.show()



if __name__ == "__main__":
    main()