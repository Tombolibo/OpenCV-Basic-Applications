import cv2
import numpy as np
import matplotlib.pyplot as plt

import chineseImgReader

imgBGR = chineseImgReader.imgRead(r'./pics/斯卡利茨.png')
imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

#灰度图
histDataofGray = cv2.calcHist([imgGray], [0], None, [256], [0,256])
imgGrayEql = cv2.equalizeHist(imgGray)
histDataofGrayEql = cv2.calcHist([imgGrayEql], [0], None, [256], [0, 256])

plt.figure('img Gray', (8,6), 100)
plt.plot(histDataofGray, color = 'r', label = 'org')
plt.plot(histDataofGrayEql, color = 'b', label = 'eql')
plt.legend()
plt.show()

cv2.imshow('img gray', imgGray)
cv2.imshow('img gray eql', imgGrayEql)
cv2.waitKey(0)

#BGR图
# histDataofB = cv2.calcHist([imgBGR], [0,1,2], None, [256,256,256], [0,256,0,256,0,256])
# print(histDataofB.shape)
#这种方式计算的是联合直方图，histData[i,j,k]表示B为iG为jR为k的像素值个数，所以是(256,256,256)

histDataofB = cv2.calcHist([imgBGR], [0], None, [256], [0,256])
histDataofG = cv2.calcHist([imgBGR], [1], None, [256], [0,256])
histDataofR = cv2.calcHist([imgBGR], [2], None, [256], [0,256])
plt.figure('img bgr', (8,6), 100)
plt.plot(histDataofB, label = 'B', color = 'b')
plt.plot(histDataofG, label = 'G', color = 'g')
plt.plot(histDataofR, label = 'R', color = 'r')
plt.legend()
plt.show()

imgBGREqlB = cv2.equalizeHist(imgBGR[:,:,0])
imgBGREqlG = cv2.equalizeHist(imgBGR[:,:,1])
imgBGREqlR = cv2.equalizeHist(imgBGR[:,:,2])
imgBGREql = np.stack([imgBGREqlB, imgBGREqlG, imgBGREqlR], axis = 2)
print(imgBGREql.shape)
histDataofBEql = cv2.calcHist([imgBGREql], [0], None, [256], [0,256])
histDataofGEql = cv2.calcHist([imgBGREql], [1], None, [256], [0,256])
histDataofREql = cv2.calcHist([imgBGREql], [2], None, [256], [0,256])

plt.figure('img bgr eql', (8,6), 100)
plt.plot(histDataofBEql, label = 'B', color = 'b')
plt.plot(histDataofGEql, label = 'G', color = 'g')
plt.plot(histDataofREql, label = 'R', color = 'r')
plt.legend()
plt.show()

cv2.imshow('img BGR org', imgBGR)
cv2.imshow('img BGR Eql', imgBGREql)
cv2.waitKey(0)