import cv2
import numpy as np

cap = cv2.VideoCapture(0)

image_x, image_y = 100,100

from keras.models import load_model
classifier = load_model('digit.h5')
classifier.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(100, 100))
       test_image = image.img_to_array(test_image)
       test_image = np.expand_dims(test_image, axis = 0)
       #result = classifier.predict(test_image)
       result = classifier.predict_classes(test_image)
       return result
      # if result[0][0] == 1:
      #        return '0'
      # elif result[0][1] == 1:
      #        return '1'
      # elif result[0][2] == 1:
      #        return '2'
      # elif result[0][3] == 1:
      #        return '3'
      # elif result[0][4] == 1:
      #        return '4'
      # elif result[0][5] == 1:
      #         return '5'
      # elif result[0][6] == 1:
      #         return '6'
      # elif result[0][8] == 1:
      #         return '7'
      # elif result[0][8] == 1:
      #         return '8'
      # elif result[0][9] == 1:
      #         return '9'
      
       
       
       


while(1):
    _,frame = cap.read()
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower = np.array([0, 48, 80], dtype = "uint8")
    #upper = np.array([20, 255, 255], dtype = "uint8")
    
    #mask = cv2.inRange(hsv, lower, upper)
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    upper_left = (50, 50)
    bottom_right = (300, 300)

    #blur = cv2.medianBlur(res,15,75,75)
    #cv2.imshow('Blur',blur)
    #cv2.imshow('Mask',mask)
    #cv2.imshow('result',res)
    #cv2.imshow('result',res)
    #edges = cv2.Canny(blur,100,200)
    #cv2.imshow('Edges',edges)
    r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 5)
    roi = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    cv2.imshow('roi',roi)

    img = cv2.resize(roi,(100,100))
    img = np.reshape(img,[1,100,100,3])
    classes = classifier.predict_classes(img)
    print(classes)

    
    img_name = "1.png"
    
    save_img = cv2.resize(roi, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    originalImage = cv2.imread('1.png')
    #grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    #(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    #cv2.imshow('Black white image', blackAndWhiteImage)
    #img_text =
    kk=predictor()
    if(cv2.waitKey(1) & 0xFF == ord('q')):
               break

cv2.destroyAllWindows()
cap.release()
