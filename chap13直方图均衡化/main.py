import cv2
import matplotlib.pyplot as plt
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/斯卡利茨.png')
imgBGR = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#使用matpoltlib绘制直方图
# plt.figure('sikalici org hist', figsize=(8,6),dpi = 100)
# plt.hist(img.ravel(), 256, label='org256')
# plt.hist(img.ravel(), 16, label='org16', alpha=0.5)

#使用cv2的calcHist计算直方图信息
histData = cv2.calcHist([img], [0], None, [256], [0,256], None, False)
#利用plt将计算的直方图信息绘制出来
plt.figure('cv2.clacHist data', (8,6), 100)
plt.plot(histData, label = 'rog')

#展示直方图
plt.legend()
plt.show()

#利用cv2.calcHist计算三个通道的直方图信息
histDataofB = cv2.calcHist([imgBGR], [0], None, [256], [0,256])
histDataofG = cv2.calcHist([imgBGR], [1], None, [256], [0,256])
histDataofR = cv2.calcHist([imgBGR], [2], None, [256], [0,256])
plt.figure('hist of BGRimg', (8,6), 100)
plt.plot(histDataofB, label = 'B', color = 'b')
plt.plot(histDataofG, label = 'G', color = 'g')
plt.plot(histDataofR, label = 'R', color = 'r')
plt.legend()
plt.show()