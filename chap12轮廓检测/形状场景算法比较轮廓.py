import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/轮廓2五角星七角星.png')
img = cv2.resize(img, None, None, 0.4, 0.4)
imgCopy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshT, img = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contoursPoly = []
for i in range(len(contours)):
    arclength = cv2.arcLength(contours[i], True)
    contoursPoly.append(cv2.approxPolyDP(contours[i], 0.001*arclength, True))
    #多边形近似的精度设置得比较低，该方法默认参数下点少了无法进行轮廓距离计算，导致报错，平常精度的0.01不够了


for i in range(len(contoursPoly)):
    cv2.drawContours(imgCopy, contoursPoly, i, (0,0,0), 3)


#形状场景距离
print('\n--------------------形状场景------------------------\n')
extractor = cv2.createShapeContextDistanceExtractor()
for i in range(len(contours)):
    for j in range(len(contours)):
        print(i,j,extractor.computeDistance(contoursPoly[i], contoursPoly[j]))


#Hausdorff距离
print('\n--------------------------hausdorff-----------------------------')
hausdorffExtrator = cv2.createHausdorffDistanceExtractor()
for i in range(len(contours)):
    for j in range(len(contours)):
        print(i,j,hausdorffExtrator.computeDistance(contoursPoly[i], contoursPoly[j]))