import numpy as np
import cv2

import chineseImgReader

imgSrc = chineseImgReader.imgRead(r'./pics/轮廓2五角星七角星.png')
imgSrc = cv2.resize(imgSrc, (int(imgSrc.shape[1]*0.5), int(imgSrc.shape[0]*0.5)))
imgGray = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)
_, imgThresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)

#获取轮廓
contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#对每个轮廓进行多边形逼近
polyList = []
eps = 0.01
for i in range(len(contours)):
    polyTemp = cv2.approxPolyDP(contours[i], eps*cv2.arcLength(contours[i], True), True)
    polyList.append(polyTemp)


#把0号画出来看看
imgSrcBack1 = imgSrc.copy()
cv2.drawContours(imgSrcBack1, polyList, 0, (0,0,0),3)

myHD = cv2.createHausdorffDistanceExtractor()

# 创建形状场景算法，形状上下文算法，计算形状之间的“距离”
for i in range(len(contours)):
    hdValue = myHD.computeDistance(polyList[0], polyList[i])  #对顶点的数量似乎又要求，每个轮廓顶点太少会报错
    print('0号轮廓与{}号轮廓距离：'.format(i), hdValue)
    if hdValue<300:
        cv2.drawContours(imgSrcBack1, polyList, i, (255,255,0), 2)

cv2.imshow('test', imgSrcBack1)
cv2.waitKey()
