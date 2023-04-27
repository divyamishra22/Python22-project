import cv2
import numpy as np
import face_recognition as face_rec
import os


def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

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

def findEncoding(images) :
    imgencodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgencodings.append(encodeimg)
    return imgencodings

EncodeList = findEncoding(studentimg)
