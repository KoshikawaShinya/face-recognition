import cv2 
import matplotlib.pyplot as plt

face_cascade_path = "haarcascade_frontalface_alt.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)

cap = cv2.VideoCapture(0)
wCam,hCam = 1280,800

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)

    if success == False:
        cap.release()
        cv2.destroyAllWindows()
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    color = (0,255,0)

    faces =face_cascade.detectMultiScale(img_gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = img[y:y+h,x:x+w]

    cv2.imshow("Image",img)
    k = cv2.waitKey(2) & 0xff
    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break