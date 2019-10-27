import traceback
import cv2
import numpy as np
import math
from numpy import loadtxt
from keras.models import load_model

cap = cv2.VideoCapture(0)

classifier = load_model('hand_gest.h5')
classifier.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

def predict_frame(image):
    cls=classifier.predict_classes(image)
    print(cls)
    

while(1):
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    fr = frame[100:400,200:500]
    cv2.rectangle(frame,(200,100),(500,400),(0,255,0),2) 
    #cv2.imshow('curFrame',frame)
    cv2.imshow('window for predicton',fr) 
    img = cv2.resize(fr,(100,100))
    img = np.reshape(img,[1,100,100,3]) 
    #classes = classifier.predict_classes(img)
    #print(classes)
    predict_frame(img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cap.release()

