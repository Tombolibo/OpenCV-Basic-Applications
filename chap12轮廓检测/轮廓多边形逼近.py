import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/轮廓2五角星七角星.png')
img = cv2.resize(img, None, None, 0.25, 0.25)
imgCopy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshT, img = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#对每个轮廓进行多边形逼近
imgPoly010 = imgCopy.copy()
imgPoly005= imgCopy.copy()
imgPoly001 = imgCopy.copy()
for i in range(len(contours)):
    #获得当前轮廓周长
    arcLength = cv2.arcLength(contours[i], True)
    approx010 = cv2.approxPolyDP(contours[i], 0.1*arcLength, True)
    approx005 = cv2.approxPolyDP(contours[i], 0.05*arcLength, True)
    approx001 = cv2.approxPolyDP(contours[i], 0.01*arcLength, True)
    cv2.drawContours(imgPoly010, [approx010], 0, [0,0,0], 2)
    cv2.drawContours(imgPoly005, [approx005], 0,[0,0,0], 2)
    cv2.drawContours(imgPoly001, [approx001], 0, [0,0,0], 2)
cv2.imshow('approx010', imgPoly010)
cv2.imshow('approx005', imgPoly005)
cv2.imshow('approx001', imgPoly001)
cv2.waitKey(0)