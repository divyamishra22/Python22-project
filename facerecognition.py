# importing libraries
import cv2 
import numpy as npy 
import face_recognition as face_rec

#function for image resize
def resize(img, size):
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension = (width,height)
    return cv2.resize(img,dimension, interpolation= cv2.INTER_AREA)



#img declaration
divyaimg= face_rec.load_image_file('images\divya.jpg')

divyaimg = cv2.cvtColor(divyaimg, cv2.COLOR_BGR2RGB)
divyaimg = resize(divyaimg, 0.50)

#face location
facelocation_divya = face_rec.face_locations(divyaimg)[0]
encode_divya = face_rec.face_encodings(divyaimg)[0]
cv2.rectangle(divyaimg,(facelocation_divya[3], facelocation_divya[0]) ,
 (facelocation_divya[1], facelocation_divya[2]), (255 , 0, 255), 3)

cv2.imshow('main_img', divyaimg)


cv2.waitKey(0)
cv2.destroyAllWindows()


