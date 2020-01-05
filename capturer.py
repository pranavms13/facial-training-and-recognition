# include standard modules
import argparse
import cv2
import time, os

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

width = cap.get(3)
height = cap.get(4)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    os.mkdir("images")
except FileExistsError:
    pass

img_dir = os.path.join(BASE_DIR, "images")

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="set person name")
parser.add_argument("-c", "--captures", help="set number of captures")

args = parser.parse_args()
person = args.name
captures = int(args.captures)

if (person and captures):
    print("Person name : %s" % person)
    i = 1
    while(i<=captures):
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=3)
        #eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=10)

        for(xf, yf, wf, hf) in faces:
            i += 1
            roi_gray = gray[yf:yf+hf, xf:xf+wf] 

            img_item = str(int(time.time())) + ".png"
            try:
                os.mkdir(img_dir + "/" + person)
            except FileExistsError:
                pass

            cv2.imwrite(img_dir + "/" + person + "/" + img_item, frame)
            time.sleep(0.25)
            fcolor = (255,0,0) #BGR
            stroke = 2
            cv2.rectangle(frame,(xf,yf),(xf+wf,yf+hf), fcolor, stroke)

            
            font = cv2.FONT_HERSHEY_DUPLEX
            name = "Capturing data for " + person
            color = (0,0,255)
            stroke = 1
            cv2.putText(frame, name, (int(width/3),yf), font, 0.5, color, stroke, cv2.LINE_AA)

        cv2.imshow('Face Recognition',frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Please Enter person name using \"-n <your name> -c <sample size>\" parameters")
    exit(0)

