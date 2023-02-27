from io import BytesIO
import threading
import time
import queue
import imageio
import mediapipe as mp
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import socket, struct
from ConvNextModel.model import GazeLSTM
from Utilites import spherical2cartesial,spherical2cartesial2
from PIL import Image
import cv2
import numpy as np

#the queue to receive the data
imageQueue = queue.Queue(maxsize=21) #maxsize=7
count = 0
count2 = 0
#add the logic for when the queue is full
#receive thread
def detectGaze(v, bbx_all):
    '''
    v = array of 7 images
    bbx_all = array of bounding boxes of face for all images
    '''
    image = v[2].copy()

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
    
    
    
    #cv2.imshow("x", image)
    #cv2.waitKey(1)
    image = cv2.resize(image,(224,224))
    out.append_data(image)
      
    return reply
#################################################################


def receive_func(client, imageQueue):
    count = 0
    while not stop_event.is_set():
        #print("Thread 2 is running")
        print("Queue Size: ",imageQueue.qsize())
        #print('x')
        #print
        try:
            #print('y')
            #start = time.time()
            #count = count+1
            dataLength = client.recv(4)
            totalDataLength = int.from_bytes(dataLength, byteorder='little')
            
            #print('EyePosition: ', eyePos)
            receivedData = b''
            
            while len(receivedData) < totalDataLength:
                packet = client.recv(totalDataLength)
                if not packet:
                    break
                receivedData+=packet
            # Load the image from the byte data
            
            imageData = BytesIO(receivedData)
            print("DataLength", totalDataLength)
            image = Image.open(imageData)
            
            print("received ", count)

            nimg = np.array(image)
            frame = cv2.cvtColor(nimg,cv2.COLOR_BGRA2RGB)#cv2.COLOR_BGR2RGB
            image = cv2.rotate(frame, cv2.ROTATE_180)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            imageQueue.put(image)
        except Exception as e:
            print(e)
            stop_event.set()
            break
    #print(f"{thread_name} stopped")

#process thread
######
#Get the face bounding boxes
#Accumulate 7 images
#Send it to the neural network 
#send output back to the client
##########

def process_func(imageQueue):
    v = []
    bbx_all = []
    count = 0
    while not stop_event.is_set():
        print("Queue Size Process Loop:", imageQueue.qsize)
        try:
            if imageQueue.qsize!=0:
                #print("Found not empty queue")
                # remove 14 items of the 21 size queue if queue is full
                if(imageQueue.full()):
                    for i in range(14):
                        imageQueue.get()

                image = imageQueue.get()
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #results = face_detection.process(image)
                v.append(image)

                height, width, channels =image.shape

                
                #xmin = (detection.location_data.relative_bounding_box.xmin)
                #ymin = (detection.location_data.relative_bounding_box.ymin)
                #widthF =(detection.location_data.relative_bounding_box.width)
                #heightF = (detection.location_data.relative_bounding_box.height)
            
                #x0,x1,y0,y1,eye = calcEyePositionBBX(0,0,width,height, width, height)# this function wont work correctly
                bbx_all.append([0,width,0,height])

                if len(v)>6: 
                    start_time = time.time()
                    detectGaze(v, bbx_all)
                    stop_time = time.time()
                    exec_time = start_time - stop_time
                    print("Execution Time GazeModel: ", exec_time)
                    print("FPS Gaze Model: ", 1/exec_time)
                    v.pop(0)
                    bbx_all.pop(0)
                else:  
                    print("Received") 
                    
                    reply = "0.0 0.0 0.0"
                    reply_len = len(reply)
                    reply_bytes = reply.encode('ascii')
                    reply_len = len(reply_bytes)
                    client.sendall(reply_len.to_bytes(4, byteorder='little'))
                    client.sendall(reply_bytes)
            
            else:
                print("Count to break: ", count)
                count = count + 1
                if count>21:
                    stop_event.set()
                    break
        except Exception as e:
            print(e)
            stop_event.set()
            break



#prepare the mediapipe functions
#removed the mediapipe functions

#load the model
image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = GazeLSTM()
model = torch.nn.DataParallel(model)
checkpoint = torch.load('ConvNextModel/model_best_Gaze360.pth.tar', map_location=torch.device('cuda'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

#setup the writer and other member variables
out = imageio.get_writer('1.mp4')

# Set up the server
hostName = socket.gethostname()
print(hostName)
host = socket.gethostbyname(hostName)
port = 8000
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.bind((host, port))
server.listen(1)
print("Server started on {}:{}".format(host, port))

# Accept connections from clients
client, address = server.accept()
print("Connected to {}".format(address))

# Receive the image data from the client

#v = []
#bbx_all =[]


# create stop event
stop_event = threading.Event()

# create first thread with args
recv_thread = threading.Thread(target=receive_func, args=(client,imageQueue))
print("Thread 1 has started successfully")
# create second thread with args
process_thread = threading.Thread(target=process_func, args=(imageQueue,))
print("Thread 2 has started successfully")
# start both threads
recv_thread.start()
process_thread.start()

# main thread code

#the code to stop the threads
while True:
    user_input = input("Press 'q' to stop threads: ")
    if user_input == 'q':
        stop_event.set()
        break

# wait for both threads to finish
recv_thread.join()
process_thread.join()

print("Both threads have finished")
