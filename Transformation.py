
import numpy as np
import math


def CreateTransformMatrix(tz, theta_x, theta_y, theta_z):
    
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    
    Rx = np.array([[1,0,0],
                   [0,np.cos(theta_x), -np.sin(theta_x)],
                   [0,np.sin(theta_x), np.cos(theta_x)]])

    Ry = np.array([[np.cos(theta_y),0,np.sin(theta_y)],
                   [0,1,0],
                   [-np.sin(theta_y),0,np.cos(theta_y)]])

    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z),0],
                   [0,0,1]]) 
    R = Rz.dot(Ry).dot(Rx)
    
    t = np.array([[tz[0]],
                  [tz[1]],
                  [tz[2]]])
    
    T = np.zeros((4,4))
    T[:3,:3]  = R
    T[:3,3] = t.ravel()
    T[3,3] = 1

    return T

def unit_vector(v):
    return v/np.linalg.norm(v)

def magv(v):
    return math.sqrt(v[0]**2+v[1]**2+v[2]**2)

def calcAngle2Vectors(v1, v2):
    dot = v1.dot(v2)

    mag1 = magv(v1)
    mag2 = magv(v2)
    return math.degrees(math.acos(dot/(mag1*mag2)))


def calcTransformation(T, v):
    #print("T: ", T.shape)
    #print('Rotation Transformation Matrix')
    #print(T[0:3,0:3])
    rotated_vector = T[0:3,0:3].dot(v[0])
    #print("Rotated Only Vector: " , rotated_vector)

    vectors_homogenous = np.column_stack((v, np.ones((len(v), 1))))
    #print('vh: ', vectors_homogenous.shape)
    vectors_transformed_homogenous = T @ vectors_homogenous.T
    #print('vth:' ,vectors_transformed_homogenous.shape)
    vectors_transformed = vectors_transformed_homogenous[:3].T
    #print('vt',vectors_transformed.shape)
    return vectors_transformed


objPos = np.asarray([10,1,0])
gazeloc = np.asarray([7.4625-5,-4.02125+7,0])
axis = np.asarray([5,7,0])
b = np.asarray([1,2,0])
#print(CreateTransformMatrix([0,0,0],0,90,0))
#print("Executing the function")

TransformationMatrix = CreateTransformMatrix(axis, 180,0,0)
#print('Transformation to be applied', TransformationMatrix)
#print("Vector to be transformed", b)
transformedGaze = calcTransformation(TransformationMatrix, [gazeloc])

transformedAxis = calcTransformation(TransformationMatrix,[axis])
print("Transformed Axis point:", transformedAxis)
dog = axis -transformedGaze[0]
objVector =  axis-objPos

print('Gaze Direction', gazeloc)
print('Object Position:', objPos)
print('Transformed Gaze: ',transformedGaze)
print('Subtracet, should be equal to 1st' , dog)

#print('Actual Angle:' ,calcAngle2Vectors(np.asarray([2,-3,0]), np.asarray([5,-6,0])))
print('Angle between 2 lines', calcAngle2Vectors(dog, objVector))