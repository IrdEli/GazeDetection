import cv2
import torch
import torchvision.transforms as transforms
import imageio
import cv2
import random
from PIL import Image
import math
import torch
import torchvision.transforms as transforms
import sys

import matplotlib.pyplot as plt

import imageio
import numpy as np
import os
from pathlib import Path
import pickle
import densepose

image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#v_reader = imageio.get_reader('video.mp4')
v=[]
cap= cv2.VideoCapture('output.mkv')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
count =0
while cap.isOpened():
  success, image = cap.read()
  #print(success)
  if success:
    v.append(image)
    #cv2.imshow('data',image)
  else:
      break
  
cap.release()

#print(len(v))
#v = list(v_reader) 
# # v is the list of all the frames think of the alternative

with Path('content/DensePoseData/ird.pkl').open('rb') as fid:
  data = pickle.load(fid)

def get_head_bbox(body_bbox, dense_pose_record):
    #Code from Detectron modifyied to make the box 30% bigger
    """Compute the tight bounding box of a binary mask."""
    # Head mask
    labels = dense_pose_record.labels.cpu().numpy()
    mask = (labels == 23) | (labels == 24)    
    if not np.any(mask):
      return None

    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    W = labels.shape[0]
    H = labels.shape[1]
    
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]

    y0 = ys[0]
    y1 = ys[-1]
    w = x1-x0
    h = y1-y0
    
    x0 = max(0,x0-w*0.15)
    x1 = max(0,x1+w*0.15)
    y0 = max(0,y0-h*0.15)
    y1 = max(0,y1+h*0.15)

    x0 += body_bbox[0]
    x1 += body_bbox[0]
    y0 += body_bbox[1]
    y1 += body_bbox[1]

    return np.array((x0, y0, x1, y1), dtype=np.float32)


def extract_heads_bbox(frame):
    N = len(frame['pred_densepose'])
    if N==0:
      return []
    bbox_list = []
    for i in range(N-1,-1,-1):
      dp_record = frame['pred_densepose'][i]
      body_bbox = frame['pred_boxes_XYXY'][i].cpu().numpy()
      bbox = get_head_bbox(body_bbox, dp_record)
      if bbox is None: 
        continue
      bbox_list.append(bbox)
    return bbox_list

names = np.array([int(Path(frame['file_name']).stem[-4:]) - 1 for frame in data], object)
bboxes = np.array([extract_heads_bbox(frame) for frame in data], object)
final_results = {n: b for n, b in zip(names, bboxes) if len(bboxes) > 0} 

#print(data[0]['pred_densepose'][0])
#print(data[0]['pred_boxes_XYXY'][0])
#print(bboxes[0][0])
#imageio.imwrite('test.png', (data[0]['pred_densepose'][1].labels.cpu().numpy() == 23).astype(float))

def compute_iou(bb1,bb2):
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    bb2_area = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    eps = 1e-8

    if iou <= 0.0  or iou > 1.0 + eps: return 0.0

    return iou
    
def find_id(bbox,id_dict):
    id_final = None
    max_iou = 0.5
    for k in id_dict.keys():
        if(compute_iou(bbox,id_dict[k][0])>max_iou): 
            id_final = k
            max_iou = compute_iou(bbox,id_dict[k][0])
    return id_final


id_num = 0
tracking_id = dict()
identity_last = dict()
frames_with_people = list(final_results.keys())

frames_with_people.sort()
for i in frames_with_people:
    speople = final_results[i]
    identity_next = dict()
    for j in range(len(speople)):
        bbox_head = speople[j]
        if bbox_head is None: continue
        id_val = find_id(bbox_head,identity_last)
        if id_val is None: 
            id_num+=1
            id_val = id_num
        #TODO: Improve eye location
        eyes = [(bbox_head[0]+bbox_head[2])/2.0, (0.65*bbox_head[1]+0.35*bbox_head[3])]
        identity_next[id_val] = (bbox_head,eyes)
    identity_last = identity_next
    tracking_id[i] = identity_last


###############################################
sys.path.append('code/')
from model import GazeLSTM
model = GazeLSTM()
model = torch.nn.DataParallel(model).cuda()
model.cuda()
checkpoint = torch.load('gaze360_model.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

###################
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
    
    gl.glEnableVertexAttribArray(0);
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 0, vertexPositions)
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
  img_buf = gl.glReadPixels(0, 0, WIDTH, HEIGHT, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
  img = np.frombuffer(img_buf, np.uint8).reshape(HEIGHT, WIDTH, 3)[::-1]
  return img

def spherical2cartesial(x): 
  output = torch.zeros(x.size(0),3)
  output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
  output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
  output[:,1] = torch.sin(x[:,1])
  return output

out = imageio.get_writer('GazeConvo.mp4',fps=fps)
print("FPS = {}".format(fps))
color_encoding = []
for i in range(1000): color_encoding.append([random.randint(0,254),random.randint(0,254),random.randint(0,254)])

W = max(int(fps//8),1)
print('W  = {}'.format(W))
for i in range(0,len(v)):
    print(i,len(v))
    
    image = v[i].copy()
    image = cv2.resize(image,(WIDTH,HEIGHT))
    image = image.astype(float)
    
    if i in tracking_id.keys():
        for id_t in tracking_id[i].keys():
            input_image = torch.zeros(7,3,224,224)
            count = 0
            for j in range(i-3*W,i+4*W,W):
                #print("I  = {}: id_T = {}, J = {}, count = {}".format(i,id_t, j, count))
                if j in tracking_id and id_t in tracking_id[j]:
                    new_im = Image.fromarray(v[j],'RGB')
                    bbox,eyes = tracking_id[j][id_t]
                else:
                    new_im = Image.fromarray(v[i],'RGB')
                    bbox,eyes = tracking_id[i][id_t]
                new_im = new_im.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
                #plt.imshow(new_im)
                #plt.savefig('CroppedImagesB/{}.jpg'.format(count*i))
                input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(new_im)))
                count = count+1
            
            bbox,eyes = tracking_id[i][id_t] 
            bbox = np.asarray(bbox).astype(int)

            # imageio.imwrite('temp.jpg', input_image[0].permute(1,2,0).cpu().numpy())

            output_gaze,_ = model(input_image.view(1,7,3,224,224).cuda())
            gaze = spherical2cartesial(output_gaze).detach().numpy()
            eyes = np.asarray(eyes).astype(float)
            print(i, gaze)
            eyes[0],eyes[1] = eyes[0]/float(v[i].shape[1]),eyes[1]/float( v[i].shape[0])

            gaze = gaze.reshape((-1))

            img_arrow = render_frame(2*eyes[0]-1,-2*eyes[1]+1,-gaze[0],gaze[1],-gaze[2],0.05)
            #print(i, img_arrow)
            binary_img = ((img_arrow[:,:,0]+img_arrow[:,:,1]+img_arrow[:,:,2])==0.0).astype(float)
            binary_img = np.reshape(binary_img,(HEIGHT,WIDTH,1))
            binary_img = np.concatenate((binary_img,binary_img,binary_img), axis=2)
            image = binary_img*image + img_arrow*(1-binary_img)
            image = image.astype(np.uint8)
            bbox[0],bbox[2] = WIDTH*bbox[0]/v[i].shape[1],WIDTH*bbox[2]/v[i].shape[1]
            bbox[1],bbox[3] = HEIGHT*bbox[1]/v[i].shape[0],HEIGHT*bbox[3]/v[i].shape[0]
            image = cv2.rectangle(image, (bbox[0],bbox[1]), (bbox[2],bbox[3]),color_encoding[min(id_t,900)])
            #print(bbox[0], bbox[1], bbox[2], bbox[3])
            image = image.astype(float)

    image = image.astype(np.uint8)
    out.append_data(image)
out.close()
