import cv2
import numpy as np
import face_recognition as face_rec
import os

path = 'images'
studentimg = []
studentName = []
myList = os.listdir(path)
print(myList)
for cl in myList :
    img = cv2.imread(f'{path}/{cl}')
    studentimg.append(img)