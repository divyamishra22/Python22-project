import cv2
import numpy as np
import face_recognition as face_rec
import os

path = 'images'
studentimg = []
studentname = []
myList = os.listdir(path)
print(myList)
for cl in myList :
    img = cv2.imread(f'{path}/{cl}')
    studentimg.append(img)
    studentname.append(os.path.splitext(cl)[0])

print(studentname)