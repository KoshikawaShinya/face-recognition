import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
wCam,hCam = 1280,800

model = cv2.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')


while True:
    success, img = cap.read()
    print(img.shape)
    img = cv2.flip(img,1)

    if success == False:
        cap.release()
        cv2.destroyAllWindows()
        break
    
    
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    