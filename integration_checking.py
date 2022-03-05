from scipy.spatial import distance as dist
import asyncio
from datetime import datetime
import random
import websockets
import cv2
import imutils
from imutils import face_utils
import dlib
import io
import base64
import requests
import numpy as np
import re, time, base64
from PIL import Image
from io import BytesIO
from imageio import imread
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import torchvision.transforms as transforms
from model_arch import Model
import pickle
import torch
import os


blinktime=[]
talktime=[]

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)
	return ear
    
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])

    mar = (A + B + C) / (2.0 * D)
    return mar


# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

increment = 0

MOUTH_AR_THRESH = 0.09
MOUTH_AR_CONSECUTIVE_FRAMES = 15
MOUTH_COUNTER=0
MOUTH_TOTAL=0


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass


left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]


kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass


# load our serialized face detector from disk
print("[INFO] loading face detector...")
detector = dlib.get_frontal_face_detector()
detector1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
protoPath = os.path.join(os.getcwd(), "deploy.prototxt")
modelPath = os.path.join(os.getcwd(), "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = Model()
model.load_state_dict(torch.load(os.path.join(os.getcwd(), "model.pth"), map_location=torch.device('cpu')))
model.eval()


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
video_capture = cv2.VideoCapture(0) 
time.sleep(3.0)


while True:
    # grab the frame from the threaded video stream and resize it
	# to have a maximum width of 500 pixels
    ret_value, frame = video_capture.read()
    thresh = frame.copy()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    ### 1 HAAR CASCADE
    # perform face detection 
    rects = detector1.detectMultiScale(gray, scaleFactor=1.05,  
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) > 1:
        print("More than one face detected. Please ensure you are in an isolated environment.")
        print("AND THEN TRY AGAIN")
        break

    # loop over the bounding boxes
    for (x, y, w, h) in rects:
        # draw the face bounding box on the image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
    
    
    ### 2 DLIB
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
            print("More than one face detected. Please ensure you are in an isolated environment.")
            print("AND THEN TRY AGAIN")
            break



    ### 3 & 4 EYE & LIP MOVEMENT
    for rect in rects:
		
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth = shape[mStart:mEnd]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        mar = mouth_aspect_ratio(mouth)
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)


        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        
        mouthHull = cv2.convexHull(mouth)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) 	    
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                if TOTAL == 3:
                    blink_flag = 1
                    print(blink_flag)
                    print(str(datetime.now()))
                #   blinktime.append(str(datetime.now()))
            # reset the eye frame counter
            COUNTER = 0

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "TALKING", (140, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            talk_flag = 1
            print(talk_flag)
            print(str(datetime.now()))
        else:
            cv2.putText(frame, "NOT TALKING", (140, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)		
            
        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
    
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break



    ### 5 EYE TRACKER DLIB
    rects = detector(gray, 1)
    for rect in rects:

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(frame, frame, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = 100
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, frame)
        contouring(thresh[:, mid:], mid, frame, True)
        # for (x, y) in shape[36:48]:
        #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



    ### 6 EYE TRACKER HAAR
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('Frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break



    ### 7 SPOOF
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

	# loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the detected bounding box does fall outside the
            # dimensions of the frame
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            # extract the face ROI and then preproces it in the exact
            # same manner as our training data
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face)
            face = face.transpose(Image.ROTATE_270)
            transform = transforms.Compose(
                [
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ]
            )

            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            face = transform(face)
            output = model(face.unsqueeze(0))
            print(output)
            output = torch.where(output < 0.25, torch.zeros_like(output), torch.ones_like(output))
            if output.item() == 0:
                label = "{}".format("Live")
                # draw the label and bounding box on the frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            else:
                label = "{}".format("Spoof")
                # draw the label and bounding box on the frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break




# Release the capture once all the processing is done. 
video_capture.release()                                  
cv2.destroyAllWindows() 
