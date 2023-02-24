
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from torch.nn.init import normal, constant
import math
import torch.utils.model_zoo as model_zoo
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from GazeModel import GazeLSTM
import mediapipe as mp
from PIL import Image
import _thread

torch.cuda.empty_cache()
#to convert spherical coordinates to cartesian
def spherical2cartesial(x): 
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])
    return output
################################
#defining the resnet model

image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



def nnTask():
  model = GazeLSTM()
  model = torch.nn.DataParallel(model)
  checkpoint = torch.load('gaze360_model.pth.tar', map_location=torch.device('cuda'))
  model.load_state_dict(checkpoint['state_dict'])

  model.eval()


  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils


  #TODO
  #1- detect the face from the video
  #2- reshape the face to fit the neural network model
  #3- Feed forward from the neural network
  #4- Convert the output from spherical to cartesian coordinates
  #5- Draw the output on the image

  imageArray = np.empty(8, dtype=np.ndarray)
  input_image = torch.zeros(7,3,224,224)
  fps = 0
  #Capture the video
  cap = cv2.VideoCapture('TestVideo.mp4') #"output.mkv"

  with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
      print('ok')
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      height, width, channels =image.shape
      #print(type(image))
      results = face_detection.process(image)

      # Capture the face
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          xmin = detection.location_data.relative_bounding_box.xmin * width
          ymin = detection.location_data.relative_bounding_box.ymin * height
          widthF = detection.location_data.relative_bounding_box.width * width
          heightF = detection.location_data.relative_bounding_box.height * height
          x0 = math.floor(max(0, xmin - widthF*0.15))
          x1 = math.floor(min(width, x0+widthF+widthF*0.15))
          y0 = math.floor(max(0, ymin - heightF*0.15))
          y1 = math.floor(min(height, y0+ heightF+ heightF*0.15))
          
          face_image = image[y0:y1, x0:x1]
          fps=fps+1
          count = 0
          #reshape the head image to feed to the neural network
          #neural network needs 7 frames of image
          
          if(fps>6):
            for j in range(fps-7,fps):
              input_image[count,:,:,:]= image_normalize(transforms.ToTensor()(transforms.Resize((224,224))(Image.fromarray(face_image))))
              #print(input_image.shape)
              count = count + 1
            #print("The Network does something")
            count = 0
          output_gaze,_ = model(input_image.view(1,7,3,224,224)) 
          #check the paper for how image is fed to the network
          
          gaze = spherical2cartesial(output_gaze).detach().numpy()
          print(gaze)
          '''
          fig = plt.figure(figsize=plt.figaspect(2.))
          fig.suptitle('A tale of 2 subplots')
          # First subplot
          ax = fig.add_subplot(2, 1, 1)
          ax.imshow(face_image)
          ax.grid(True)
          ax.set_ylabel('The image')
          # Second subplot
          ax = fig.add_subplot(2, 1, 2, projection='3d')
          ax.quiver(0,0,0, gaze[0][0],gaze[0][1],gaze[0][2])
          ax.set_zlim(-1, 0)
          ax.set_xlabel('x')
          ax.set_xlabel('y')
          ax.set_xlabel('z')
          ax.set_ylim(-1,1)
          ax.set_xlim(0,1)
          plt.pause(0.2)
          '''
          '''
          #convert to cartezian coordinates
          fig, (ax1, ax2) = plt.subplots(ncols=2)
          ax1 = plt.subplot(311)
          ax1.imshow(face_image)
          ax2 = plt.subplot(312, projection='3d')
          ax2.quiver(0,0,0, gaze[0][0],gaze[0][1],gaze[0][2])
          plt.show()
          '''



          #print(x0, x1, y0, y1)
          #draw the detected face
          #mp_drawing.draw_detection(image, detection)

          #3d printing the gaze along side the frames
          
          #fig.close()
          
      cv2.imshow('MediaPipe Face Detection', image)
    
    
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
  cap.release()
  return


def plotTask():
  fig = plt.figure()
  fig.show()
  ax = plt.axes(projection='3d')
  zline = np.linspace(0, 15, 1000)
  xline = np.sin(zline)
  yline = np.cos(zline)
  ax.plot3D(xline, yline, zline, 'gray')
  return


nnTask()

'''

        fig = plt.figure()
        #fig.show()
        ax = plt.axes(projection='3d')
        zline = np.linspace(0, 15, 1000)
        xline = np.sin(zline)
        yline = np.cos(zline)
        ax.plot3D(xline, yline, zline, 'gray')

'''