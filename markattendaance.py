import cv2
import numpy as np
import face_recognition as face_rec
import os

path = 'images'
studentImg = []
studentName = []
myList = os.listdir(path)
print(myList)