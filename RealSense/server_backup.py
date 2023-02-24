import socket, cv2, pickle, struct
import realsense
from realsense_depth import *
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
#socket create
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = socket.gethostbyname(host_name)

print('Host_IP: ', host_ip)
port = 9998
socket_address = (host_ip, port)

#Socket Bind
server_socket.bind(socket_address)

#Socket Listen
server_socket.listen(5)
print('Listening At:', socket_address)

#Socket Accept
def get_eye_gaze_position(results):
    for detection in results.detections:
                #mp_drawing.draw_detection(image, detection)
                xmin = (int)(detection.location_data.relative_bounding_box.xmin*640)
                xmax = (int)( xmin + detection.location_data.relative_bounding_box.width*640)
                ymin = (int)(detection.location_data.relative_bounding_box.ymin*480)
                ymax = (int)(detection.location_data.relative_bounding_box.height*480)
                #print(detection.location_data.relative_keypoints[1])
                x1 = (int)(detection.location_data.relative_keypoints[0].x*640)
                y1= (int)(detection.location_data.relative_keypoints[0].y*480)
                
                x2 = (int)(detection.location_data.relative_keypoints[1].x*640)
                y2= (int)(detection.location_data.relative_keypoints[1].y*480)
                
                eye1 = (x1,y1)
                eye2 = (x2,y2)
                #print(xmin,ymin,xmax, ymax)
                #cv2.circle(image, eye1,4,(0,255,0))
                #cv2.circle(image, eye2,4,(0,255,0))
                #e1_depth = depth[eye1[1],eye1[0]]
                #e2_depth = depth[eye2[1],eye2[0]]
                x = (int)((x1+x2)/2)
                y = (int)((y1+y2)/2)
                eyes =(x,y)
                #cv2.circle(image, eyes,4,(0,255,0))
                print('Eye Coordinats',eyes)
                eyes_depth = depth[eyes[1], eyes[0]] 
                position = rs.rs2_deproject_pixel_to_point(d_intrin, [eyes[0],eyes[1]], eyes_depth)
                positionM = []
                for i in position:
                    positionM.append(i/1000)
                print('Eye Position:',positionM)
    return positionM

dc = DepthCamera()

while True:
    print('Start')
    client_socket, addr = server_socket.accept()
    print('Got Connection From: ', addr)

    
    if client_socket:
        #vid = cv2.VideoCapture(0)
        
        #ret, depth, color = dc.get_frame()
        while(True):
            with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
                ret, depth, frame,d_intrin, c_intrin = dc.get_frame()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                position = [0,0,0]
                if results.detections:
                    position = get_eye_gaze_position(results)
                #img, frame = vid.read()
                #print(img)
                #create a tuple pickle

                temp = (depth, frame, position)
                print(frame.shape)
                a = pickle.dumps(temp)
                message = struct.pack('Q', len(a))+a
                client_socket.sendall(message)
                print('frame sent ...')
                cv2.imshow('Transmitting Video', frame)
                #################################
            
            '''
            server_socket.listen()
            client_socket, client_address = server_socket.accept()
            data =b''
            payload_size = struct.calcsize('Q')
            while len(data)<payload_size:
                packet = server_socket.recv(4*1024)
                if not packet: break
                data+=packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack('Q', packed_msg_size)[0]

            while len(data)<msg_size:
                data+= server_socket.recv(4*1024)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            frame = pickle.loads(frame_data)
            print('Received Frame')
            '''
            
            ###############################    
            key = cv2.waitKey(1) &0xFF
            if key == ord('q'):
                client_socket.close()



'''
import socket


#socket creation
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#socket bind
server.bind(('localhost', 6400))#127.0.0.1

#socket listen
server.listen()

#socket accpet
client_socket, client_address = server.accept()

file = open('server_image.pdf', 'wb')

#handle client
image_chunk = client_socket.recv(2048) #takes buffersize to receive from stream

while image_chunk:
    print('receiving')
    file.write(image_chunk)
    image_chunk = client_socket.recv(2048)

file.close()

#close client
client_socket.close()
'''