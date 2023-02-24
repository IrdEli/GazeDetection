import torch
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt



def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output

def spherical2cartesial2(x):
    output = torch.zeros(x.size(0),3)
    output[:,0] = torch.sin(x[:,0])* torch.cos(x[:,1])
    output[:,1] = torch.sin(x[:,1])* torch.sin(x[:,0])
    output[:,2] = torch.cos(x[:,0])
    return output


def cartesial2spherical(x):
    output = [0,0,0]
    #print("input:", x)
    if(type(x[0]) is tuple):
        #print(x[0][0],x[1][0],x[2][0])
        output[0] = math.sqrt(float(x[0][0])**2+float(x[1][0])**2+float(x[2][0])**2)
        output[1] = math.degrees(math.atan2(x[1][0],x[0][0]))
        if(output[0] == 0):
            output[2]=0
        else:
            output[2] = math.acos(x[2][0]/output[0])
    else: 
        output[0] = math.sqrt(float(x[0])**2+float(x[1])**2+float(x[2])**2)
        output[1] = math.degrees(math.atan2(x[1],x[0]))
        if(output[0] == 0):
            output[2]=0
        else:
            output[2] = math.degrees(math.acos(x[2]/output[0]))
    return output

def writeOnImage(image, texts, position):
    '''
        takes image, text(as array) and position(as tuple) to write the text on the image
    
    '''
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.5
    fontColor =(0,0,255)
    thickness = 1
    lineType = 2
    count = 1
    for text in texts:
        cv2.putText(image, text, (position[0], position[1]+count*40), font, fontScale, fontColor,thickness, lineType)
        count = count+1    
    return image

def calcEyePositionBBX(xmin, ymin, widthF,heightF, width, height):
    x0 = math.floor(max(0, xmin - widthF*0.5)*width)
    x1 = math.floor(min(width, x0 + (widthF+widthF)*width))
    y0 = math.floor(max(0, ymin - heightF*0.5)*height)
    y1 = math.floor(min(height, y0+(heightF+ heightF)*height))
    eye = [(x0+x1)/2.0, (0.65*y0+0.35*y1)]
    
    return x0,x1,y0,y1,eye

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

def calcAngle2Vectors(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2[0])
    #print(v1_u,v2_u)
    return np.arccos(np.clip(np.dot(v1_u,v2_u), -1.0,1.0))


def calcTransformation(T, v):
    #print("T: ", T.shape)
    vectors_homogenous = np.column_stack((v, np.ones((len(v), 1))))
    #print('vh: ', vectors_homogenous.shape)
    vectors_transformed_homogenous = T @ vectors_homogenous.T
    #print('vth:' ,vectors_transformed_homogenous.shape)
    vectors_transformed = vectors_transformed_homogenous[:3].T
    #print('vt',vectors_transformed.shape)
    return vectors_transformed

