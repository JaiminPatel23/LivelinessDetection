from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
#import argparse
import imutils
import time
import dlib
import cv2
import random
import io
import requests
import re, base64
import matplotlib.pyplot as plt
from datetime import datetime

talktime=[]
    
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])

    mar = (A + B + C) / (2.0 * D)
    return mar
 
MOUTH_AR_THRESH = 0.09
MOUTH_AR_CONSECUTIVE_FRAMES = 15

# initializing the frame counters and the total number of lip movements
MOUTH_COUNTER=0
MOUTH_TOTAL=0


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# grab the indexes of the facial landmarks for the upper and
# lower lip, respectively
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


print("[INFO] starting video stream thread...")

vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

# loop over frames from the video stream
while True:
	
	if fileStream and not vs.more():
		break

	
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		mouth = shape[mStart:mEnd]
		mar = mouth_aspect_ratio(mouth)

		# compute the convex hull for the mouth (upper and lower lip), 
		# then visualize
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		
		if mar > MOUTH_AR_THRESH:
			cv2.putText(frame, "TALKING", (140, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			talk_flag = 1
			print(talk_flag)
			print(str(datetime.now()))
		else:
			cv2.putText(frame, "NOT TALKING", (140, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)	

		
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
