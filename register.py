import cv2
import numpy as np
import os
import uuid

NAME="Tanisha Lakhani"
os.makedirs(os.path.join('application_data', 'verification_images', NAME), exist_ok=True)

from cvzone.FaceMeshModule import FaceMeshDetector
detector = FaceMeshDetector(maxFaces=1)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') 

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    bfaces = face_cascade.detectMultiScale(frame, 1.2, 10, minSize=(64,64))
    for (x, y, w, h) in bfaces: 
        facess = frame[y+50:y + 300, x+50:x + 300] 
        # cv2.rectangle(frame, (x, y), (x+w, y+h),  
        #             (0, 0, 255), 2) 
    
    cv2.imshow('Verification', frame)
    
    # Verification trigger
    if cv2.waitKey(2) in [27, ord('a'), ord('A')]:
        # Create the unique file path 
        print(f"Capturing images for {NAME}...")
        imgname = os.path.join('application_data', 'verification_images', NAME, '{}_{}.jpg'.format(NAME, uuid.uuid1()))
        cv2.imwrite(imgname, facess)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()