#from re import I
import torch
import torch.nn as nn
import torch.nn.parallel
from ctypes import sizeof
import cv2
import mediapipe as mp
import torchvision.transforms as transforms
import math
from GazeModel import GazeLSTM
from PIL import Image
import imageio
import random
import matplotlib.pyplot as plt
#####################################################################################################################
import numpy as np
import json
import moviepy.editor as mvp
#from google.colab import files
import ctypes
import ctypes.util
from ctypes import pointer
import os
from lucid.misc.gl.glcontext import create_opengl_context
import OpenGL.GL as gl

WIDTH, HEIGHT = 960, 720
create_opengl_context((WIDTH, HEIGHT))

gl.glClear(gl.GL_COLOR_BUFFER_BIT)

from OpenGL.GL import shaders
WIDTH, HEIGHT = 960, 720
create_opengl_context((WIDTH, HEIGHT))

vertexPositions = np.float32([[-1, -1], [1, -1], [-1, 1], [1, 1]])
VERTEX_SHADER = shaders.compileShader("""
#version 330
layout(location = 0) in vec4 position;
out vec2 UV;
void main()
{
  UV = position.xy*0.5+0.5;
  gl_Position = position;
}
""", gl.GL_VERTEX_SHADER)

FRAGMENT_SHADER = shaders.compileShader("""
#version 330
out vec4 outputColor;
in vec2 UV;

uniform sampler2D iChannel0;
uniform vec3 iResolution;
vec4 iMouse = vec4(0);
uniform float iTime = 0.0;
uniform float xpos = 100.0;
uniform float ypos = 0.1;
uniform float size = 10.0;

uniform float vdir_x = -1.0;
uniform float vdir_y = -1.0;
uniform float vdir_z = 1.0;



const vec3 X = vec3(1., 0., 0.);
const vec3 Y = vec3(0., 1., 0.);
const vec3 Z = vec3(0., 0., 1.);

// YOURS

const float Z_NEAR = 1.0;
const float Z_FAR  = 400.0;

const float EPSILON = 0.01;

const float FOCAL_LENGTH = 30.0;
const vec3 EYE_LOOK_POINT = vec3(0, 0, 5);

const vec3 WHITE = vec3(1, 1, 1);
const vec3 BLACK = vec3(0, 0, 0);
const vec3 RED =   vec3(1, 0, 0);
const vec3 GREEN = vec3(0, 1, 0);
const vec3 BLUE =  vec3(0, 0, 1);

const vec3 TOP_BG_COLOR = WHITE;
const vec3 BOT_BG_COLOR = GREEN;



const vec3 AMBIANT_COLOR = WHITE;
const vec3 SPECULAR_COLOR = WHITE;

const float AMBIANT_RATIO = 0.3;
const float DIFFUSE_RATIO = 0.8;
const float SPECULAR_RATIO = 0.4;
const float SPECULAR_ALPHA = 5.;

const vec3 LIGHT_DIRECTION = normalize(vec3(1, -1, -1));

vec2 normalizeAndCenter(in vec2 coord) {
    return (2.0 * coord - iResolution.xy) / iResolution.y;
}

vec3 rayDirection(vec3 eye, vec2 uv) {    
    vec3 z = normalize(eye - EYE_LOOK_POINT);
    vec3 x = normalize(cross(Y, z));
    vec3 y = cross(z, x);
    
    return normalize(
          x * uv.x 
        + y * uv.y 
        - z * FOCAL_LENGTH);
}

//
// Rotations
//

vec3 rotX(vec3 point, float angle) {
    mat3 matRotX = mat3(
        1.0, 0.0, 0.0, 
        0.0, cos(angle), -sin(angle), 
        0.0, sin(angle), cos(angle));
    return matRotX * point;
}

vec3 rotY(vec3 point, float angle) {
    mat3 matRotY = mat3( 
        cos(angle*0.5), 0.0, -sin(angle*0.5),
        0.0, 1.0, 0.0, 
        sin(angle*0.5), 0.0, cos(angle*0.5));
    return matRotY * point;
}

vec3 rotZ(vec3 point, float angle) {
    mat3 matRotZ = mat3(
        cos(angle*0.1), -sin(angle*0.1), 0.0, 
        sin(angle*0.1), cos(angle*0.1), 0.0,
    	0.0, 0.0, 1.0);
    return matRotZ * point;
}

//
// Positioning
//

vec3 randomOrtho(vec3 v) {
    if (v.x != 0. || v.y != 0.) {
    	return normalize(vec3(v.y, -v.x, 0.));
    } else {
    	return normalize(vec3(0., v.z, -v.y));
    }
} 

vec3 atPosition(vec3 point, vec3 position) {
	return (point - position);
}

vec3 atCoordSystem(vec3 point, vec3 center, vec3 dx, vec3 dy, vec3 dz) {
	vec3 localPoint = (point - center);
    return vec3(
        dot(localPoint, dx),
        dot(localPoint, dy),
        dot(localPoint, dz));
}

vec3 atCoordSystemX(vec3 point, vec3 center, vec3 dx) {
    vec3 dy = randomOrtho(dx);
    vec3 dz = cross(dx, dy);
    
    return atCoordSystem(point, center, dx, dy, dz);
}

vec3 atCoordSystemY(vec3 point, vec3 center, vec3 dy) {
    vec3 dz = randomOrtho(dy);
    vec3 dx = cross(dy, dz);
    
    return atCoordSystem(point, center, dx, dy, dz);
}

vec3 atCoordSystemZ(vec3 point, vec3 center, vec3 dz) {
    vec3 dx = randomOrtho(dz);
    vec3 dy = cross(dz, dx);
    
    return atCoordSystem(point, center, dx, dy, dz);
}

//
// Shapes
//

float capsule(vec3 coord, float height, float radius)
{
    coord.y -= clamp( coord.y, 0.0, height );
    return length( coord ) - radius;
}

float roundCone(vec3 coord, in float radiusTop, float radiusBot, float height)
{
    vec2 q = vec2( length(coord.xz), coord.y );
    
    float b = (radiusBot-radiusTop)/height;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));
    
    if( k < 0.0 ) return length(q) - radiusBot;
    if( k > a*height ) return length(q-vec2(0.0,height)) - radiusTop;
        
    return dot(q, vec2(a,b) ) - radiusBot;
}

//
// Boolean ops
//

vec4 shape(float dist, vec3 color) {
	return vec4(color, dist);
}

vec4 join(vec4 shape1, vec4 shape2) {
    if (shape1.a < shape2.a) {
    	return shape1;
    } else {
    	return shape2;
    }
}

vec4 join(vec4 shape1, vec4 shape2, vec4 shape3) {
    return join(join(shape1, shape2), shape3);
}

vec4 join(vec4 shape1, vec4 shape2, vec4 shape3, vec4 shape4) {
    return join(join(shape1, shape2, shape3), shape4);
}



//
// Scene
// x range: 355
// y rangeL 205



vec4 dist(in vec3 coord) {
    vec3 V_y = normalize(vec3(vdir_x,vdir_y,vdir_z));
    
    vec3 V_x = randomOrtho(V_y);
        
    vec3 V_z = cross(V_x,V_y);
    
    vec3 pos = vec3(xpos,ypos,0);
        
    vec3 offset = pos+V_y*size*4.0;
    vec3 ARM_2_X = V_x;
    vec3 ARM_2_Y = V_y;
    vec3 ARM_2_Z = V_z;
    vec3 NOZE_O = vec3(0,11,0);
    vec3 NOZE_X = -V_x;
    vec3 NOZE_Y = V_y;
    vec3 NOZE_Z = V_z;

    vec4 noze = shape(roundCone(atCoordSystem(coord, offset, NOZE_X, NOZE_Y, NOZE_Z), size*0.05, size*0.4, size*2.0), RED);
    vec4 rightArm = shape(capsule(atCoordSystem(coord, pos, ARM_2_X, ARM_2_Y, ARM_2_Z), size*5., size*0.1), RED);
    return join(noze, rightArm);
}

//
//
//

bool rayMarching(in vec3 startPoint, in vec3 direction, out vec3 lastPoint, out vec3 color) {
    lastPoint = startPoint;
    for (int i = 0; i < 50; ++i) {
        vec4 d = dist(lastPoint);
        if (d.a < EPSILON) {
            color = d.xyz;
            return true;
        } else {
            lastPoint += d.a * direction;
        }
        if (lastPoint.z < -Z_FAR) {
            break;
        }
    }    
    return false;
}

vec3 norm(in vec3 coord) {
	vec3 eps = vec3( EPSILON, 0.0, 0.0 );
	vec3 nor = vec3(
	    dist(coord+eps.xyy).a - dist(coord-eps.xyy).a,
	    dist(coord+eps.yxy).a - dist(coord-eps.yxy).a,
	    dist(coord+eps.yyx).a - dist(coord-eps.yyx).a);
	return normalize(nor);
}


vec3 cellShadingObjColor(vec3 point, vec3 ray, vec3 objColor) {
    vec3 n = norm(point);
    
    float diffuseValue = max(dot(-LIGHT_DIRECTION, n), 0.);
    float specularValue = pow(max(dot(-reflect(LIGHT_DIRECTION, n), ray), 0.), SPECULAR_ALPHA);        
    return AMBIANT_COLOR * AMBIANT_RATIO
        + objColor * DIFFUSE_RATIO * diffuseValue
        + SPECULAR_COLOR * SPECULAR_RATIO * specularValue;
}


vec3 computeColor(vec2 fragCoord) {
    vec2 uv = normalizeAndCenter(fragCoord);
    vec3 eye = vec3(0, 0, 20);
        
    vec3 ray = rayDirection(eye, uv);
    
    vec3 intersection;
    vec3 color;
    bool intersected = rayMarching(eye, ray, intersection, color);
    if (intersected) {
    	return cellShadingObjColor(intersection, ray, color);
    } else {
    	return vec3(0,0,0);
    }
}

//#define SUPERSAMPLING
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{    
    #ifdef SUPERSAMPLING
    fragColor = vec4(0);
    float count = 0.;
    for(float dx=-0.5; dx<=0.5; dx+=0.5) {
    	for(float dy=-0.5; dy<=0.5; dy+=0.5) {
            fragColor += vec4(computeColor(fragCoord + vec2(dx, dy)), 1.0);
            count += 1.;
        }
    }
    
    fragColor /= count;
    
    #else
    fragColor = vec4(computeColor(fragCoord),1.0);
    #endif
} 



void main()
{
    mainImage(outputColor, UV*iResolution.xy);
}

""", gl.GL_FRAGMENT_SHADER)

shader = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

xpos = gl.glGetUniformLocation(shader, 'xpos')
ypos = gl.glGetUniformLocation(shader, 'ypos')

vdir_x = gl.glGetUniformLocation(shader, 'vdir_x')
vdir_y = gl.glGetUniformLocation(shader, 'vdir_y')
vdir_z = gl.glGetUniformLocation(shader, 'vdir_z')

arrow_size = gl.glGetUniformLocation(shader, 'size')

res_loc = gl.glGetUniformLocation(shader, 'iResolution')

def render_frame(x_position,y_position,vx,vy,vz,asize):
  gl.glClear(gl.GL_COLOR_BUFFER_BIT)
  with shader:

    x_position = x_position*0.89
    y_position = y_position*0.67
    gl.glUniform1f(xpos, x_position)
    gl.glUniform1f(ypos, y_position)

    gl.glUniform1f(vdir_x, vx)
    gl.glUniform1f(vdir_y, vy)
    gl.glUniform1f(vdir_z, vz)
    gl.glUniform1f(arrow_size, asize)

    gl.glUniform3f(res_loc, WIDTH, HEIGHT, 1.0)
    
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, vertexPositions)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
  img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
  img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]
  return img
  ##########################################################
  #################################################################################################

image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


model = GazeLSTM()
model = torch.nn.DataParallel(model)
checkpoint = torch.load('gaze360_model.pth.tar', map_location=torch.device('cuda'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output


def detectGaze(v, bbx_all,eyes ,noPPF):
    #print(len(v), len(bbx_all), len(noPPF))
    image = v[3].copy()
    input_image = torch.zeros(7,3,224,224)
    '''
    Loop through all the frames
    Check how many people were in the frame
    Detect the gaze for every single person
    return the gaze vector

    Check the logic again to deal with multiple people. Use some sort of id to keep track of the same person in the scene
    '''
    count = 0
    for i in range(-3, 4):
        print(i,count)
        for j in range(noPPF[i]):
            face_image = v[i][bbx_all[i+j][2]:bbx_all[i+j][3], bbx_all[i+j][0]:bbx_all[i+j][1]]
            input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(Image.fromarray(face_image))))
            count = count+1
    output_gaze, _=model(input_image.view(1,7,3,224,224).cuda())
    gaze = spherical2cartesial(output_gaze).detach().numpy()
    #print(i, gaze)
    eye[0] = eye[0]/float(v[i].shape[1])
    eye[1] = eye[1]/float(v[i].shape[0])

    gaze = gaze.reshape((-1))
    
    #draw the arrow on the image
    img_arrow = render_frame(2*eye[0]-1,-2*eye[1]+1,-gaze[0],gaze[1],-gaze[2],0.05)
    #print(i, img_arrow)
    binary_img = ((img_arrow[:,:,0]+img_arrow[:,:,1]+img_arrow[:,:,2])==0.0).astype(float)
    binary_img = np.reshape(binary_img,(HEIGHT,WIDTH,1))
    binary_img = np.concatenate((binary_img,binary_img,binary_img), axis=2)
    image = binary_img*image + img_arrow*(1-binary_img)
    image = image.astype(np.uint8)
    image = cv2.rectangle(image, (bbx_all[3][0],bbx_all[3][2]), (bbx_all[3][1],bbx_all[3][3]),color_encoding[900])
    out.append_data(image)
    #plt.imshow(image)
    #cv2.imshow('Output', image)
    #cv2.waitKey(1000)
    return

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils



input_image = torch.zeros(7,3,224,224)

out = imageio.get_writer('2.mp4')#,fps=cv2.CAP_PROP_FPS)
color_encoding = []
for i in range(1000): color_encoding.append([random.randint(0,254),random.randint(0,254),random.randint(0,254)])
# Create a VideoCapture object and read from input fil
#cap = cv2.VideoCapture("output.mkv") 
#count =0
#v=[]

v=[]
bbx_all = []
eyes = []
noPPF = []#track number of people in each frame
#cap= cv2.VideoCapture('output.mkv')
#length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#fps = cap.get(cv2.CAP_PROP_FPS)

cap = cv2.VideoCapture(0)


count = 0
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    
  #print(success)
    if success:
      count = count + 1
      if count>100:
        break

      #image.flags.writeable = False 
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = face_detection.process(image)
      if results.detections:
        image = cv2.resize(image, (WIDTH,HEIGHT))
        v.append(image)
        noPPF.append(len(results.detections))
        height, width, channels =image.shape
        for detection in results.detections:
          xmin = (detection.location_data.relative_bounding_box.xmin)
          ymin = (detection.location_data.relative_bounding_box.ymin)
          widthF =(detection.location_data.relative_bounding_box.width)
          heightF = (detection.location_data.relative_bounding_box.height)
          
          #print(xmin, widthF, ymin, heightF)
          #mp_drawing.draw_detection(image, detection)
          x0 = math.floor(max(0, xmin - widthF*0.5)*width)
          x1 = math.floor(min(width, x0 + (widthF+widthF)*width))
          y0 = math.floor(max(0, ymin - heightF*0.5)*height)
          y1 = math.floor(min(height, y0+(heightF+ heightF)*height))
          eye = [(x0+x1)/2.0, (0.65*y0+0.35*y1)]
          bbx_all.append([x0,x1,y0,y1])
          eyes.append(eye)


          if len(v)>6:
            #run the algorithm after the first 6 frames
            detectGaze(v, bbx_all, eyes, noPPF)
            v.pop(0)
                #print('x')
            
                #print(noPPF)
            for items in range(noPPF[0]):
                bbx_all.pop(0)
                eyes.pop(0)
            noPPF.pop(0)
        #print('x')
       
          mp_drawing.draw_detection(image, detection)
    #cv2.imshow('data',image)
        if cv2.waitKey(500) & 0xFF == 27:
             break
    else:
        print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
        continue
cap.release()
out.close()
print("Finished Writing the arrays on the face!")


cap = cv2.VideoCapture('2.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
height, width, channels =image.shape
print('New Video',length, fps, height, width)
