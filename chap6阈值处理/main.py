import cv2
import numpy as np

img = cv2.imread(r'./pics/poker.png')
img = cv2.resize(img, None, None, 0.5, 0.5)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 大津法自适应全局阈值处理
_, imgThresh = cv2.threshold(img, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(_, imgThresh.shape)

#自适应阈值处理，适合明暗差异较大图像
imgAdaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101,0)

cv2.imshow('imgThresh', imgThresh)
cv2.imshow('imgAdaptive', imgAdaptive)
cv2.waitKey(0)