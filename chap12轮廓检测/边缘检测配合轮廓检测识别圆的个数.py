import cv2
import numpy as np

import chineseImgReader

imgSrc = chineseImgReader.imgRead(r'./pics/相近颜色重叠圆形.png') #7个圆
imgGray = cv2.cvtColor(imgSrc, cv2.COLOR_BGRA2GRAY)

#canny边缘检测
maskCanny = cv2.Canny(imgGray, 30,60)

#对canny结果进行轮廓识别
contours, hierarchy = cv2.findContours(maskCanny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imgSrc, contours, -1, (0,125,255),5)


cv2.imshow('test', imgSrc)
cv2.imshow('canny', maskCanny)
cv2.waitKey()