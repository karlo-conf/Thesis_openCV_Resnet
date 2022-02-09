
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2


model = keras.models.load_model('C:\\Users\\Karl\\Desktop\\Thesis_openCV_Resnet\\GarbageSeperateModel')
class_names = ['biodegradable', 'nonbiodegradable', 'recyclable']

cam = cv2.VideoCapture(1)

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
    output_class = class_names[np.argmax(pred)]
    accuracy = np.argmax(pred)
    true_accuracy = pred[0][accuracy]
    print("The item should be in the", output_class, "bin")
    new_img = cv2.putText(
      img = img_read,
      text = "Prediction : {}".format(output_class),
      org = (1,20),
      fontFace = cv2.FONT_HERSHEY_PLAIN,
      fontScale = 1.5,
      color = (255, 255, 255),
      thickness = 2,
    )
    new_img = cv2.putText(
      img = img_read,
      text = "Accuracy : {}".format(true_accuracy),
      org = (1,50),
      fontFace = cv2.FONT_HERSHEY_PLAIN,
      fontScale = 1.5,
      color = (255, 255, 255),
      thickness = 2,
    )

    cv2.imshow("Prediction", new_img)




cam.release()
cv2.destroyAllWindows()