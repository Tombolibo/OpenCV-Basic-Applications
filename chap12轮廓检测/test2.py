import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/轮廓.png')
t = 0.5
img = cv2.resize(img,(int(img.shape[1]*t), int(img.shape[0]*t)))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, imgThresh = cv2.threshold(imgGray, 127,255,cv2.THRESH_BINARY_INV)

contours, here = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(here[0][0])
for i in range(len(contours)):
    cv2.drawContours(img, contours,i,(0,0,255),3,8,here[0],2)

cv2.imshow('test', img)
cv2.waitKey()