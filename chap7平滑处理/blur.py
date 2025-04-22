import cv2
import numpy as np

img = cv2.imread(r'./pics/qb.png')
img = cv2.resize(img, None, None, 0.5, 0.5)

#均值滤波
imgBlur = cv2.blur(img, (10,10))

#方框滤波，求和
imgBox = cv2.boxFilter(img, -1,(3,3),normalize=0)

#高斯滤波
imgGaussian = cv2.GaussianBlur(img, (5,5),0,None,0)

#中值滤波
imgMedianBlur = cv2.medianBlur(img, 15)

#双边滤波
imgBilateral = cv2.bilateralFilter(img, -1,50,10)

cv2.imshow('imgSrc', img)
cv2.imshow('blur', imgBlur)
cv2.imshow('box', imgBox)
cv2.imshow('gaussian', imgGaussian)
cv2.imshow('median', imgMedianBlur)
cv2.imshow('bilateral', imgBilateral)
cv2.waitKey(0)
