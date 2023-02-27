import time
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
import socket, struct
import traceback
import numpy as np
import moviepy.editor as mvp
import matplotlib.pyplot as plt
from Utilites import spherical2cartesial, spherical2cartesial2,calcAngle2Vectors,calcEyePositionBBX,calcTransformation,CreateTransformMatrix,unit_vector,cartesial2spherical,writeOnImage



def detectGaze(v, bbx_all,eyeToDraw ,noPPF, eye_position, x_rot, y_rot,z_rot):
   
    image = v[2].copy()
    eye_position = np.asarray(eye_position)
    input_image = torch.zeros(7,3,224,224)
    count = 0
    for i in range(-3, 4):
        face_image = v[i][bbx_all[i][2]:bbx_all[i][3], bbx_all[i][0]:bbx_all[i][1]]
        input_image[count,:,:,:] = image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(Image.fromarray(face_image))))
        count = count+1
    output_gaze, _= model(input_image.view(1,7,3,224,224).cuda())
    gaze = spherical2cartesial(output_gaze).detach().numpy() # this function is fucked up need to verify this

    reply = np.array2string(np.asarray(gaze))
    print('Reply', reply)
    reply_len = len(reply)
    reply_bytes = reply.encode('ascii')
    reply_len = len(reply_bytes)
    client.sendall(reply_len.to_bytes(4, byteorder='little'))
    client.sendall(reply_bytes)
    
    
    out.append_data(image)
      
    return
#################################################################

while(True):
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
    checkpoint = torch.load('C:/Users/irdal/OneDrive/Desktop/ConvNextGazeSystem/ConvNextModel/model_best_Gaze360.pth.tar', map_location=torch.device('cuda'))
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
                print(i, length)
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
            
            
            #############################################################
            #-To code the actual logic of processing the image
            # 1- Get the rotation of the head in the frame and save it to the array
            
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
                    
                    v.append(image)
                    
                    eyePos7Frames.append(eyePos)
                    noPPF.append(len(results.detections))
                    height, width, channels =image.shape
                    
                    for detection in results.detections:
                        
                        xmin = (detection.location_data.relative_bounding_box.xmin)
                        ymin = (detection.location_data.relative_bounding_box.ymin)
                        widthF =(detection.location_data.relative_bounding_box.width)
                        heightF = (detection.location_data.relative_bounding_box.height)
                    
                        x0,x1,y0,y1,eye = calcEyePositionBBX(xmin,ymin,widthF,heightF, width, height)
                        bbx_all.append([x0,x1,y0,y1])
                        eyeToDraw.append(eye) 
                        if len(v)>6: 
                            start_time = time.time()
                            detectGaze(v, bbx_all, eyeToDraw[2], noPPF, eyePos7Frames[0],x_rot, y_rot,z_rot)
                            end_time = time.time()
                            execution_time = end_time - start_time
                            print("Execution Time = ", execution_time)
                            print("FPS: ", 1/execution_time)
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
        except Exception as e:
            print("Caught the exception:", e, traceback.format_exc())
            out.close()
    # Close the connection
            cv2.destroyAllWindows()
            client.close()
            server.close()
            break
            



