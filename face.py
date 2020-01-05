import numpy as np
import cv2
import pickle 

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    oglabels = pickle.load(f)
    labels = {v:k for k,v in oglabels.items()}


cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=3)
    #eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=10)

    for(xf, yf, wf, hf) in faces:
        roi_gray = gray[yf:yf+hf, xf:xf+wf] 

        img_item = "myimg.png"
        cv2.imwrite(img_item, frame)
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 30:
            print(id_)
            print(labels[id_], conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (int(xf+xf/3),yf), font, 1, color, stroke, cv2.LINE_AA)

        #print(xf,yf,wf,hf)
        
        


        fcolor = (255,0,0) #BGR
        stroke = 2
        cv2.rectangle(frame,(xf,yf),(xf+wf,yf+hf), fcolor, stroke)

    # for (xe, ye, we, he) in eyes:
    #     roi_gray = gray[ye:ye+he, xe:xe+he]
    #     ecolor = (0,255,0) #BGR
    #     stroke = 1
    #     cv2.rectangle(frame,(xe,ye),(xe+we,ye+he), ecolor, stroke)

    cv2.imshow('Face Recognition',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()