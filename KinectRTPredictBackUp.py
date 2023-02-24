#from re import I
import torch
import torch.nn as nn
import torch.nn.parallel
from ctypes import sizeof
import cv2
import mediapipe as mp
import torchvision.transforms as transforms
import math
from ConvNextModel.model import GazeLSTM
from PIL import Image
import imageio
import random
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.interpolate import interp1d
import socket, pickle, struct
import traceback
#####################################################################################################################
import numpy as np
import json
import moviepy.editor as mvp
#from google.colab import files
import ctypes
import ctypes.util
from ctypes import pointer
import os
#from lucid.misc.gl.glcontext import create_opengl_context
#import OpenGL.GL as gl
from itertools import permutations
import matplotlib.pyplot as plt
from Utilites import spherical2cartesial, spherical2cartesial2,calcAngle2Vectors,calcEyePositionBBX,calcTransformation,CreateTransformMatrix,unit_vector,cartesial2spherical,writeOnImage


def createmap(map,kinectLocation, kinectRotation, kinectTransformToWCMatrix, matrix, eye_position, gazeWCVec, VecRAGWC, gaze):
    epWC = calcTransformation(kinectTransformToWCMatrix, calcTransformation(matrix, [eye_position]))
    print(kinectLocation, eye_position[0])
    cv2.line(map,(int(kinectLocation[0]),int(kinectLocation[1])),(int(epWC[0][0]),int(eye_position[0][1])),(255,255,255))
    return map

def detectGaze(v, bbx_all,eyeToDraw ,noPPF, eye_position, x_rot, y_rot,z_rot):
   
    image = v[2].copy()
    eye_position = np.asarray(eye_position)
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
        
            face_image = v[i][bbx_all[i][2]:bbx_all[i][3], bbx_all[i][0]:bbx_all[i][1]]
            input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(Image.fromarray(face_image))))
            count = count+1
    output_gaze, _= model(input_image.view(1,7,3,224,224).cuda())
    gaze = spherical2cartesial(output_gaze).detach().numpy() # this function is fucked up need to verify this
    
    #print("Received")
    reply = np.array2string(np.asarray(gaze))
    print('Reply', reply)
    reply_len = len(reply)
    reply_bytes = reply.encode('ascii')
    reply_len = len(reply_bytes)
    client.sendall(reply_len.to_bytes(4, byteorder='little'))
    client.sendall(reply_bytes)
    '''
    calculate transformed gaze
    object location
    '''
    #vector between eye
    x_fix = 0
    y_fix = 0
    z_fix = 0
    if(count2<10):
        x_fix = -x_rot
        y_fix = -y_rot
        z_fix = -z_rot
    
    matrix = CreateTransformMatrix(eye_position[0],x_rot+x_fix+180,y_rot+y_fix+180,z_rot+z_fix+180)
    print("Gaze Vector", unit_vector(gaze[0]))
    
    gazeWCVec = calcTransformation(matrix, gaze)
    #vector between the robot and the eyeaxis translated position
    VecRAGWC = eye_position-robApos
    
    error = calcAngle2Vectors(gazeWCVec, VecRAGWC)
    
   
    print("Gaze Vector in WC : ", unit_vector(gazeWCVec))
    print("Eye Position in WC: ", eye_position)
    print("EyePosition from Transform: ", unit_vector(calcTransformation(matrix,[[0,0,0]])[0]) )
    #print('Reconfirming Gaze Vector: ',np.asarray(gazeWCVec)-np.asarray(eyePosWC) )
    print("Vector between Robot and eye ", unit_vector(VecRAGWC))
    #print("in degrees:" , cartesial2spherical(unit_vector(gaze[0])))
    print("ANgle between the 2: ", math.degrees(calcAngle2Vectors(gazeWCVec, VecRAGWC)))
    #sct.set_array(eye_position[0][0])
    ax.scatter(eye_position[0][0],eye_position[0][1],eye_position[0][2])
    
    #plt.show()
    #plt.pause(0.1)
    #plt.close()
    
    #plt.waitforbuttonpress()
   # plt.savefig('DataVisualize/{}.jpg'.format(count2))
    #print("Ratio between two outputs: ", gazeWC[0]/gaze[0][0], gazeWC[1]/gaze[0][1],gazeWC[2]/gaze[0][2])
    
    
    look = -1
    if(np.degrees(error-30)*180/10<5):
        cv2.circle(image, (50,50), 20, (0,0,255))
       
    #gazeWorldCoordinateSph = [gazeWorldCoordinateSph[0],gazeWorldCoordinateSph[1]-60,gazeWorldCoordinateSph[2]-90+gaze_deg[1]]
    ####################
    #draw the arrow on the image
    #print(eyeToDraw,)
    eyeToDraw[0] = eyeToDraw[0]/float(v[2].shape[1])
    eyeToDraw[1] = eyeToDraw[1]/float(v[2].shape[0])
    gaze = gaze.reshape((-1))
    #img_arrow = render_frame(2*eyeToDraw[0]-1,-2*eyeToDraw[1]+1,-gaze[0],gaze[1],-gaze[2],0.05)
    #binary_img = ((img_arrow[:,:,0]+img_arrow[:,:,1]+img_arrow[:,:,2])==0.0).astype(float)
    #binary_img = np.reshape(binary_img,(HEIGHT,WIDTH,1))
    #binary_img = np.concatenate((binary_img,binary_img,binary_img), axis=2)
    #image = binary_img*image + img_arrow*(1-binary_img)
    #image = image.astype(np.uint8)
    
    #confirm if looking at the position of the robot
    look = -1

    #Write the information on the image to understand the output
    inter = (interp1d([0,90], [0,180]))
        
    reconfirmedGaze =  gazeWCVec-np.asarray(eye_position[0])
    #print(reconfirmedGaze)
    texts = ['Gaze Cartesian Local: '+ np.array2string(np.asarray(gaze)),
             'Gaze Spherical Local:' +np.array2string(np.asarray(cartesial2spherical([gaze[0],gaze[1],gaze[2]]))),
             'Eye Position Global: '+ np.array2string(eye_position),
             'Gaze Cartesian Global ' + np.array2string(np.asarray(gazeWCVec)),
             'Vector from Eye to Robot'+np.array2string(np.asarray(VecRAGWC)),
             'Reconfirming gaze' + np.array2string(reconfirmedGaze),
             'Reconfirmed Spherical Gaze' + np.array2string(np.asarray(cartesial2spherical(reconfirmedGaze[0])))
             ]
    image = writeOnImage(image,texts, [30,30])
   
    map = np.zeros((500, 500, 3), np.uint8)
    #map = createmap(map, kinectLocation, kinectRotation, kinectTransformToWCMatrix, matrix, eye_position[0], gazeWCVec, VecRAGWC, gaze)
    offset = 250
    
    out.append_data(image)
      
    return
#################################################################
fig, ax = plt.subplots()
#plot = plt.plot()
ax = plt.axes(projection = '3d')

#sct = ax.scatter([0,0,0],[0,0,0],[0,0,0])

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

WorldCoordinate = np.asarray([0, 0, 0])
kinectLocation =  np.asarray([250, 250, 10])
kinectRotation = np.asarray([0,0,0])

kinectTransformToWCMatrix =CreateTransformMatrix(kinectLocation, kinectRotation[0],kinectRotation[1],kinectRotation[2])
color_encoding = []
for i in range(1000): color_encoding.append([random.randint(0,254),random.randint(0,254),random.randint(0,254)])


image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = GazeLSTM()
model = torch.nn.DataParallel(model)
checkpoint = torch.load('ConvNextModel/model_best_Gaze360.pth.tar', map_location=torch.device('cuda'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()


robApos = np.asarray([0 ,0, -400])
robBpos = [[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]]


        
out = imageio.get_writer('1.mp4')
 # Set up the server
hostName = socket.gethostname()
print(hostName)
host = socket.gethostbyname(hostName)
port = 9000
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.bind((host, port))
server.listen(1)
print("Server started on {}:{}".format(host, port))

# Accept connections from clients
client, address = server.accept()
print("Connected to {}".format(address))
gif = []
# Receive the image data from the client
count = 0
count2 = 0
#To store the rotation of head in last 7 frames
x_rot = []
y_rot = []
z_rot = []
v = []
eyePos7Frames = []
eyeToDraw =[]
bbx_all =[]
noPPF = []

while(True):
    try:
        count = count+1
        dataLength = client.recv(4)
        totalDataLength = int.from_bytes(dataLength, byteorder='little')
        #print("DataLength", totalDataLength)
        eyePos = [0,0,0]
        headRot = [0,0,0]
        for i in range(6):
            length = client.recv(4) 
            #print(i, length)
            if(i<3):
                eyePos[i] = struct.unpack('<f', length)[0]
            else:
                headRot[i-3] = struct.unpack('<f', length)[0]
        #print('EyePosition: ', eyePos)
        receivedData = b''
        while len(receivedData) < totalDataLength:
            receivedData += client.recv(totalDataLength)
        # Load the image from the byte data
        imageData = BytesIO(receivedData)
        image = Image.open(imageData)
        print("received ", count)
        nimg = np.array(image)
        frame = cv2.cvtColor(nimg,cv2.COLOR_BGRA2RGB)#cv2.COLOR_BGR2RGB
        image = cv2.rotate(frame, cv2.ROTATE_180)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if(count==1):
            cv2.imwrite('x.jpg', image)
        #v.copy(frame)
        #############################################################
            #-To code the actual logic of processing the image
        # 1- Get the rotation of the head in the frame and save it to the array
        #x,y,z = getFaceRot(frame)
        #print(eyePos, headRot)
        x_rot.append(headRot[0])
        y_rot.append(headRot[1])
        z_rot.append(headRot[2])

        
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            #print(results.detections)
            
            if results.detections:
                count2 = count2+1
                print("Frames Processed: ", count2)
                print("Frames Skipped: ", count-count2)
                #image = cv2.resize(image, (960,720))
                v.append(image)
                eyePos = calcTransformation(kinectTransformToWCMatrix, [eyePos])
                eyePos7Frames.append(eyePos)
                noPPF.append(len(results.detections))
                height, width, channels =image.shape
                
                for detection in results.detections:
                    #Detect the head rectangle of the person
                    xmin = (detection.location_data.relative_bounding_box.xmin)
                    ymin = (detection.location_data.relative_bounding_box.ymin)
                    widthF =(detection.location_data.relative_bounding_box.width)
                    heightF = (detection.location_data.relative_bounding_box.height)
                    
                    #calculate the position of the eye using the above bounding box used to drwa the axis
                    x0,x1,y0,y1,eye = calcEyePositionBBX(xmin,ymin,widthF,heightF, width, height)
                    bbx_all.append([x0,x1,y0,y1])
                    eyeToDraw.append(eye)
                    
                    #if received 7 images detect the eyegaze, draw the arrow, calculate the coordinate transformations and everything
                    #print("Images to Process",len(v))
                    
                    
                    if len(v)>6:
                        #print("Entering to Gaze Detection")
                        #print(headRot)
                        #avgEyePos = [0,0,0]
                        xavg = 0
                        yavg = 0
                        zavg = 0
                        xRavg = 0
                        yRavg = 0
                        zRavg = 0
                        for items in eyePos7Frames[0]:
                            xavg = items[0]+ xavg
                            yavg = items[1]+ yavg
                            zavg = items[2]+ zavg
                        for i in range(len(x_rot)):
                            xRavg = xRavg + x_rot[i]
                            yRavg = yRavg + y_rot[i]
                            zRavg = zRavg + z_rot[i]
                        detectGaze(v, bbx_all, eyeToDraw[2], noPPF, [[xavg/7, yavg/7,zavg/7]],xRavg/7, yRavg/7,zRavg/7)
                        v.pop(0)
                        eyePos7Frames.pop(0)
                        x_rot.pop(0)
                        y_rot.pop(0)
                        z_rot.pop(0)
                        for items in range(noPPF[0]):
                            bbx_all.pop(0)
                            eyeToDraw.pop(0)
                        noPPF.pop(0)
                    else:    
                        
                        reply = "0.0 0.0 0.0"
                        reply_len = len(reply)
                        reply_bytes = reply.encode('ascii')
                        reply_len = len(reply_bytes)
                        client.sendall(reply_len.to_bytes(4, byteorder='little'))
                        client.sendall(reply_bytes)
            else:
                #print("Received")
                reply = "0.0 0.0 0.0"
                reply_len = len(reply)
                reply_bytes = reply.encode('ascii')
                reply_len = len(reply_bytes)
                client.sendall(reply_len.to_bytes(4, byteorder='little'))
                client.sendall(reply_bytes)
        #frame = writeOnImage(frame, [str(x)+str(y)+str(z)], [30,30])     
        #########################################################
        # Do something with the image, for example save it to disk
       
        #out.append_data(frame)
        #cv2.imshow("X", ocvim)
        #cv2.waitKey(1)
        #image.save("Video\{}.jpg".format(count))
    except Exception as e:
        print("Caught the exception:", e, traceback.format_exc())
        out.close()
# Close the connection
        cv2.destroyAllWindows()
        client.close()
        server.close()
        break
            



