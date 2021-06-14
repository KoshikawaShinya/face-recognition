import os
import cv2
import matplotlib.pyplot as plt

print("Please Enter Your Name.")
name = input()
path_template = '../images/{}/'.format(name)

if not os.path.exists(path_template):
    os.makedirs(path_template)

cap = cv2.VideoCapture(0)
wCam,hCam = 1280,800
cnt = 0

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    print(img.shape)

    if success == False:
        cap.release()
        cv2.destroyAllWindows()
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=2, minSize=(80, 80))
    for (x, y, w, h) in face:
        cut_face = img[y:y+h, x:x+w]
        cv2.imwrite(path_template+str(cnt)+'.jpg', cut_face)
        img = cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0), 2)
        cnt += 1
    
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    
