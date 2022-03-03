# import the necessary packages
from matplotlib import pyplot as plt
from imutils import face_utils
import numpy as np
import random
#import argparse
import imutils
import dlib
import cv2
import sys

### Facial landmarks with dlib, OpenCV, and Python

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
video_capture = cv2.VideoCapture(0) 

while True: 
    # grab the frame from the video stream, resize it, and convert it
    # to grayscale
    ret_value, frame = video_capture.read() 
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if(i < 1):
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        
        else:
            print("More than 1 face detected. Please ensure you are in an isolated environment.")
            print("AND THEN TRY AGAIN")


# Release the capture once all the processing is done. 
video_capture.release()                                  
cv2.destroyAllWindows() 
