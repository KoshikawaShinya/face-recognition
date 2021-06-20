import cv2

def cut(img, face_cascade, rectangle=False):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=2, minSize=(80, 80))
    for (x, y, w, h) in face:
        cut_face = img[y:y+h, x:x+w]
        if rectangle:
            img = cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0), 2)
        return cut_face
    