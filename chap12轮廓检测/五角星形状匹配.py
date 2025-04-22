import numpy as np
import cv2

import chineseImgReader

imgSrc = chineseImgReader.imgRead(r'./pics/轮廓2五角星七角星.png')
imgSrc = cv2.resize(imgSrc, (int(imgSrc.shape[1]*0.5), int(imgSrc.shape[0]*0.5)))

imgGray = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)
_, imgThresh = cv2.threshold(imgGray, 127,255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

pattenID = 5

# for i in range(len(contours)):
#     shapeMatchRTV = cv2.matchShapes(contours[pattenID], contours[i],cv2.CONTOURS_MATCH_I1,0)
#     print(shapeMatchRTV)
#     if shapeMatchRTV<0.25:
#         cv2.drawContours(imgSrc, contours, i, (0,0,0), 3)
# cv2.drawContours(imgSrc, contours, pattenID, (255,0,0),10)
# #这matchShapes结果就很他妈令人无语，说是可以最好先做个多边形拟合
# cv2.imshow('src', imgSrc)
# cv2.waitKey(0)

#进行一次多边形逼近再进行形状匹配
imgSrcBackup = imgSrc.copy()
eps = 0.01
ploylist = []
for i in range(len(contours)):
    ployTemp = cv2.approxPolyDP(contours[i], eps*cv2.arcLength(contours[i], True), True)
    ploylist.append(ployTemp)
for i in range(len(contours)):
    matchShape = cv2.matchShapes(ploylist[pattenID], ploylist[i], cv2.CONTOURS_MATCH_I1, 0)
    print(matchShape)
    if matchShape<0.37:
        cv2.drawContours(imgSrcBackup, ploylist, i, (0,0,0), 3)
cv2.imshow('test', imgSrcBackup)
cv2.waitKey()
