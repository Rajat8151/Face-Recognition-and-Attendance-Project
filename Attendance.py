# Import Required Libraries
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# Import our Image
path  = 'resources'
image = []
className = []
mylist = os.listdir(path)
print(mylist)
for cls in mylist:
    currImg= cv2.imread(f'{path}/{cls}')
    image.append(currImg)
    className.append(os.path.splitext(cls)[0]) # to remove jpg from the name
print(className)



# Find Encodings
def findWNCODINGS(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findWNCODINGS(image)
print("Encoding complete")



# Initialzing Webcam
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceCURR = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS,faceCURR)

    for encodeFace,FaceLoc in zip(encodeCurrFrame,faceCURR):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        # Face Detection Rectangle Assign
        if matches[matchIndex]:
            name= className[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = FaceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(className[matchIndex])

    cv2.imshow("Webcam",img)
    cv2.waitKey(1)