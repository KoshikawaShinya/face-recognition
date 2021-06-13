import cv2
import torch
import time
import numpy as np
from efficientnet_pytorch import EfficientNet
from model import EaseClassificationModel

H = 64
W = 64

cap = cv2.VideoCapture(0)
model_1 = EfficientNet.from_pretrained('efficientnet-b3')
model_2 = EaseClassificationModel(H, W)


while True:
    success, img = cap.read()
    #print(img.shape)
    time.sleep(1e-3)
    imgRe = cv2.resize(img, (H, W))
    imgRGB = cv2.cvtColor(imgRe, cv2.COLOR_BGR2RGB)
    imgRGB = imgRGB.reshape(1, H, W, 3).transpose(0, 3, 1, 2)
    imgRGB = torch.tensor(imgRGB).float()
    output = model_2(imgRGB)
    #print(output)

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)


    if key == ord('q'):
        break