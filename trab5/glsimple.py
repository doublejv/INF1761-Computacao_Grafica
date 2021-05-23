from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import sys


brass=[[0.33, 0.22, 0.03, 1], [0.78, 0.57, 0.11, 1], [0.99, 0.91, 0.81, 1], [27.8]]

def setMaterial(material):
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material[0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material[1])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,material[2])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, material[3])

def displayCall( ):
    glClearColor(0.4,0.4,0.4,1)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    setMaterial(brass)
    glutSolidCube(1)
    glFlush()
# REDISPLAY callback (when canvas apear) 

def reshapeCall(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45,4./3,0.5,10)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(1.5,1.5,1.2, 0,0,0, 0,0,1)
# RESHAPE Callback (when change size, or is created (change from zero to its size))

def keyboardCall( bkey,  x,  y):
    # Convert bytes object to string 
    key = bkey.decode("utf-8")
    # save canvas in an image file
    if key=='p' or key=='s'or key=='w':
        save_frame_buffer()
    # Allow to quit by pressing 'Esc' or 'q'
    if key == chr(27):
        sys.exit()
    if key == 'q':
        print("Bye!")
        sys.exit()
# KEYBOARD Callback

def  initLight( ):
    position=[ 0.5,2.,0.,1.]
    low = [0.2,0.2,0.2,1]
    white =[1,1,1,1]
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glLightfv(GL_LIGHT0, GL_AMBIENT, low)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white)
    glLightfv(GL_LIGHT0, GL_SPECULAR, white)
    glLightfv(GL_LIGHT0, GL_POSITION, position)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

def save_frame_buffer(format="PNG"):
    x, y, width, height = glGetDoublev(GL_VIEWPORT)
    width, height = int(width), int(height)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), data)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("canvas.png", format)

def main():
    glutInit() # /* Inicializando a GLUT */ 
    glutInitDisplayMode(GLUT_RGB|GLUT_DEPTH) 
    glutInitWindowSize(640, 480) 
    glutCreateWindow("glSimple")
    glutDisplayFunc(displayCall)
    glutReshapeFunc(reshapeCall)
    glutKeyboardFunc(keyboardCall)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    initLight()    # Inicializando a luz e o material

    glutMainLoop() # GLUT main loop (control is now with GLUT) 


if __name__ == "__main__":
    main()

