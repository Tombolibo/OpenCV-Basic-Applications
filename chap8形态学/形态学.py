import cv2
import numpy as np

img = cv2.imread(r'./pics/coins.jpg')
img = cv2.resize(img, None, None, 0.5, 0.5)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshT, img = cv2.threshold(imgGray, 75, 255, cv2.THRESH_BINARY_INV)

#获得结构元
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

#腐蚀
imgErode = cv2.morphologyEx(img, cv2.MORPH_ERODE, k, None, (-1,-1), 1)
#膨胀
imgDilate = cv2.morphologyEx(img, cv2.MORPH_DILATE, k, None, (-1,-1), 1)
#开运算
imgOpen = cv2.morphologyEx(img, cv2.MORPH_OPEN, k, None, (-1,-1), 1)
#闭运算
imgClose = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, None, (-1, -1), 1)
#顶帽
imgTophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, k, None, (-1,-1), 1)
#黑帽
imgBlackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k, None, (-1,-1), 1)
#形态学梯度
imgG = cv2. morphologyEx(img, cv2.MORPH_GRADIENT, k, None, (-1,-1), 1)

cv2.imshow('imgGray', img)
cv2.imshow('erode', imgErode)
cv2.imshow('dilate', imgDilate)
cv2.imshow('open', imgOpen)
cv2.imshow('close', imgClose)
cv2.imshow('tophat', imgTophat)
cv2.imshow('blackhat', imgBlackhat)
cv2.imshow('g', imgG)
cv2.waitKey(0)