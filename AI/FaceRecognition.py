import cv2
import torch
import time
import numpy as np
from model import EaseClassificationModel
from utils.cut_face import cut
import requests
# access_tokenは自分で取得したもの
url = 'https://notify-api.line.me/api/notify'
access_token = 'Your token'
headers = {'Authorization': 'Bearer' + ' ' + access_token}

H = 64
W = 64

timeP=0
timeC=0

i = 0
path = './memo.txt'
with open(path,mode='r') as f:
    data = f.readlines()
    for datam in data:
        data[i] = datam.strip().split()
        i+=1

cap = cv2.VideoCapture(0)
model_2 = EaseClassificationModel(H, W)
model_2.load_state_dict(torch.load('models/0_cpu.pth'))

face_cascade = cv2.CascadeClassifier('utils/haarcascade_frontalface_default.xml')

while True:
    success, img = cap.read()
    #print(img.shape)
    time.sleep(1e-3)
    img_cut = cut(img, face_cascade, rectangle=True)
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
            if(timeP == 0):
                timeP = time.time()
            timeC = time.time()
            if (timeC - timeP >= 3.0):
                for i in out:
                    if out == i:
                        message = data[i][1]+'さんこんにちは'
                        payload = {'message':message}
                        r = requests.post(url, headers=headers, data=payload,)
                timeP = 0
                timeC = 0
    else:
        timeP = 0
        timeC = 0


    cv2.imshow("Image",img)
    key = cv2.waitKey(1)


    if key == ord('q'):
        break