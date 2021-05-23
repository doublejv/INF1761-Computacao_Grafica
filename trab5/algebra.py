import numpy as np
import math
#from math import sin, cos, sqrt, radians
import sys 

TOL = sys.float_info.epsilon

# Vector algebra
def vector(x,y,z):
    return np.array([x,y,z],dtype=np.float64)

def angle(v1,v2):
    num = dot(v1,v2)
    den = norm(v1)*norm(v2)
    return np.arccos(num/den)*180/np.pi if den>TOL else 0
  
def norm(v):
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def unitary(v):
    s = norm(v)
    if (s>TOL):
        return v/s
    else:
        return None

def dot(u,v):
    return (u[0]*v[0]+u[1]*v[1]+u[2]*v[2])

def reflect(v,n):
    r = 2*dot(v,n)*n-v
    return r

def cross(u,v):
    return vector(u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0])

# Projective and homogeneous vectors
def pvector(x,y,z,w):
    return np.array([x,y,z,w],dtype=np.float64)

def to_cartesian(vet4):
    if vet4[3]>TOL:
        return vector(vet4[0]/vet4[3],vet4[1]/vet4[3],vet4[2]/vet4[3])
    else:
        print("w=0")
        return None

def to_projective(v):
    return pvector(v[0],v[1],v[2],1)

# Matrix
def identity_matrix():
    return np.eye(4,dtype=np.float64)

def translation_matrix(tx,ty,tz):
    T = np.eye(4,dtype=np.float64)
    T[0,3]=tx
    T[1,3]=ty
    T[2,3]=tz
    return T

def scale_matrix(sx,sy,sz):
    S = np.eye(4,dtype=np.float64)
    S[0,0]=sx
    S[1,1]=sy
    S[2,2]=sz
    return S

def rotatation_matrix(ang, ex,ey,ez):
    size = math.sqrt(ex*ex+ey*ey+ez*ez)
    R = np.eye(4,dtype=np.float64)
    if size>TOL:
        ang = math.radians(ang)
        sin_a = math.sin(ang)
        cos_a = math.cos(ang)
        ex /= size
        ey /= size
        ez /= size
        R[0,0]  = cos_a + (1 - cos_a)*ex*ex    #Linha 1
        R[0,1]  = ex*ey*(1 - cos_a) - ez*sin_a
        R[0,2]  = ez*ex*(1 - cos_a) + ey*sin_a

        R[1,0]  = ex*ey*(1 - cos_a) + ez*sin_a
        R[1,1]  = cos_a + (1 - cos_a)*ey*ey
        R[1,2]  = ey*ez*(1 - cos_a) - ex*sin_a

        R[2,0]  = ex*ez*(1 - cos_a) - ey*sin_a
        R[2,1]  = ey*ez*(1 - cos_a) + ex*sin_a
        R[2,2] = cos_a + (1 - cos_a)*ez*ez
    
    return R

def change_basis_matrix(xe, ye, ze):
    R = np.eye(4,dtype=np.float64)
    R[0,:3] = xe 
    R[1,:3] = ye
    R[2,:3] = ze
    return R

def transf4x3(M,v):
    x = M[0,0]*v[0]+M[0,1]*v[1]+M[0,2]*v[2]+M[0,3]
    y = M[1,0]*v[0]+M[1,1]*v[1]+M[1,2]*v[2]+M[1,3]
    z = M[2,0]*v[0]+M[2,1]*v[1]+M[2,2]*v[2]+M[2,3]
    return vector(x,y,z)
