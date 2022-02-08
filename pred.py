from turtle import clear
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


model = keras.models.load_model('C:\\Users\\Karl\\Desktop\\Windows Form\\GarbageSeperateModel')
class_names = ['biodegradable', 'nonbiodegradable', 'recyclable']

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

image_name = 0

while True:
  ret, frame = cam.read()
  if not ret:
        print("failed to grab frame")
        break
  
  cv2.imshow("test", frame)

  keypressed = cv2.waitKey(1)

  if keypressed % 256 == 27:
    #ESC pressed
    print('ESC pressed Exiting')
    break
  elif keypressed % 256 == 32:
    image_name_save = '{}.jpg'.format(image_name)
    img_captured = cv2.imwrite(image_name_save, frame) #Write image to disk

    img_read = cv2.imread('{}.jpg'.format(image_name)) # Read the writed image
    img_resize = cv2.resize(img_read,(255,255)) #Resize image to 255
    img_expanded = np.expand_dims(img_resize, axis=0) #Expand the dimension of the image

    print(img_expanded.shape)

    pred = model.predict(img_expanded)
    print(pred)




cam.release()
cv2.destroyAllWindows()