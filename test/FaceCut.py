import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    #print(img.shape)
    time.sleep(1e-3)

    #print(output)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)


    if key == ord('q'):
        break