import pyrealsense2
import cv2
from realsense_depth import *
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        dc = DepthCamera()
        #point = (400,300)
        ret, depth, color,d_intrin,c_intrin = dc.get_frame()
        if not ret:
            print("Ignoring empty frame")
            continue
        image = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
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
                cv2.circle(image, eyes,4,(0,255,0))
                print(eyes)
                eyes_depth = depth[eyes[1], eyes[0]] 
                e1_point = rs.rs2_deproject_pixel_to_point(d_intrin, [eye1[0],eye1[1]], eyes_depth)
                print(e1_point)
                print("Eye1:",depth[eye2[1], eye2[0]])
                print("Eye2:",depth[eye1[1], eye1[0]])
                
                #cv2.circle(image, (xmin,ymin),4,(0,0,255))
                #cv2.circle(image, (xmin,ymax),4,(0,0,255))
                #cv2.circle(image, (xmax,ymin),4,(0,0,255))
                #cv2.circle(image, (xmax,ymax),4,(0,0,255))
    # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
        #cv2.circle(color, point, 4,(0,0,255))
        #dist = depth[point[1], point[0]]
        #print(dist)
        #cv2.imshow('Depth Frame:', depth)
        #cv2.imshow('Color Frame:', color)