
import cv2, socket, pickle, struct

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.0.19'
port = 9998

client_socket.connect((host_ip, port))

data = b""
payload_size = struct.calcsize('Q')
while True:
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024) #4kb packet
        if not packet: break
        data+=packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack('Q', packed_msg_size)[0]
    
    while len(data)<msg_size:
        data += client_socket.recv(4*1024)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data)    
    cv2.imshow('Received', frame)
    key = cv2.waitKey(1) &0xFF
    if key == ord('q'):
        break
client_socket.close()
'''
import socket

#socket create
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#socket connect
client.connect(('localhost', 6400))#127.0.0.1

file = open('OpenGaze__ICCV_2019_Sup_.pdf', 'rb')

image_data = file.read(2048)
#client send/receive
while image_data:
    client.send(image_data)
    image_data = file.read(2048)

#client close
file.close()
client.close()
'''