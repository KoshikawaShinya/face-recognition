import cv2
import torch
import time
import numpy as np
from model import EaseClassificationModel
from utils.cut_face import cut

H = 64
W = 64

cap = cv2.VideoCapture(0)
model_2 = EaseClassificationModel(H, W)
model_2.load_state_dict(torch.load('models/0_cpu.pth'))

face_cascade = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')

while True:
    success, img = cap.read()
    #print(img.shape)
    time.sleep(1e-3)
    img_cut = cut(img, face_cascade, rectangle=False)
    if img_cut is not None:
        imgRe = cv2.resize(img_cut, (H, W))
        imgRGB = cv2.cvtColor(imgRe, cv2.COLOR_BGR2RGB)
        imgRGB = imgRGB.reshape(1, H, W, 3).transpose(0, 3, 1, 2)
        with torch.no_grad():
            imgRGB = torch.tensor(imgRGB).float()
            output = model_2(imgRGB)
            out = np.argmax(output.numpy(), 1)

            cv2.putText(img, str(out), (40, 70), cv2.FONT_HERSHEY_COMPLEX,
                        3, (255, 0, 255), 3)
    #print(output)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)


    if key == ord('q'):
        break