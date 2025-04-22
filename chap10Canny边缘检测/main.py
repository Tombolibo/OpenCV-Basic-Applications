import cv2
import numpy as np

from chineseImgReader import imgReader

img = imgReader.imgRead(r'./pics/斯卡利茨.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

t, imgThresh = cv2.threshold(img,-1,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(t)

imgCanny = cv2.Canny(img,200,350)
imgCannyOtus = cv2.Canny(img,0.5*t,t)

cv2.imshow('img', img)
cv2.imshow('canny',imgCanny)
cv2.imshow('cannyOTST', imgCannyOtus)
cv2.waitKey()