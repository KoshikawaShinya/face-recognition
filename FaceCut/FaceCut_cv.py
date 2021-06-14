import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
wCam,hCam = 1280,800

name = input("名前を入力してね:")
path = "../test.txt"
i=0
f1 = open(path)
data = f1.readlines()

for datam in data:
    data[i] = datam.strip()
    num = int(data[0])
    i+=1
print(int(num))

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)

    if success == False:
        cap.release()
        cv2.destroyAllWindows()
        break

    detector = dlib.get_frontal_face_detector()
    # RGB変換 (opencv形式からskimage形式に変換)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # frontal_face_detectorクラスは矩形, スコア, サブ検出器の結果を返す
    dets, scores, idx = detector.run(img_rgb, 0)
    for det in dets:
        cv2.rectangle(img, (det.left(), det.top()), (det.right(), det.bottom()), (0, 0, 255))
        imgC = img[det.top()+2 : det.bottom() , det.left()+2 : det.right()]
        cv2.imwrite('../face/'+name+str(num)+'.jpg',imgC)
        num+=1

    cv2.imshow("Image",img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
with open(path,mode='w') as f:
    f.write("{}".format(num))
    f.write("\n")
f1.close()