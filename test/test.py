import cv2
import torch
import time
import numpy as np
from efficientnet_pytorch import EfficientNet

cap = cv2.VideoCapture(0)
model = EfficientNet.from_pretrained('efficientnet-b3')

while True:
    success, img = cap.read()
    img = cv2.resize(img, (320, 320))
    print(img.shape)
    time.sleep(1e-3)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = imgRGB.reshape(1, 320, 320, 3).transpose(0, 3, 1, 2)
    imgRGB = torch.tensor(imgRGB).float()
    output = model(imgRGB)
    print(output)
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break