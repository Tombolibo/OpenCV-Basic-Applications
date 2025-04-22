import cv2
import numpy as np

import chineseImgReader

imgSrc = chineseImgReader.imgRead(r'./pics/轮廓.png')
imgSrc = cv2.resize(imgSrc, (int(imgSrc.shape[1]*0.5), int(imgSrc.shape[0]*0.5)))
imgGray = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)
_, imgThresh = cv2.threshold(imgGray,127,255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contoursPattenID = 3

#计算轮廓的特征矩（轮廓矩），包括空间矩、中心矩（空间矩减去质心）、归一化中心矩（除以面积），归一化中心矩的线性组合叫做Hu矩
m_ofContours3 = cv2.moments(contours[contoursPattenID])
print('cv2.moments()结果类型为：', type(m_ofContours3))
print(m_ofContours3)

#计算某个轮廓的面积：cv2.contoursArea
area_ofContours3 = cv2.contourArea(contours[contoursPattenID])
print("轮廓面积：", area_ofContours3) #计算结果和m00一样

#计算某个轮廓的长度
length_ofContours3 = cv2.arcLength(contours[contoursPattenID], True)
print('计算长度类型: ', type(length_ofContours3))
print('长度值：', length_ofContours3)

#计算轮廓的Hu矩
Hu_ofContours3 = cv2.HuMoments(cv2.moments(contours[contoursPattenID]))
print('Hu矩类型：', type(Hu_ofContours3))
print('Hu矩值：', Hu_ofContours3)

#cv2利用matchShapes，通过计算两个轮廓的Hu矩或者两个灰度图的Hu矩，进行形状匹配，返回
# ID1 = 3
# ID2 = 2
# c1Andc2MatchShapeRtval = cv2.matchShapes(contours[ID1], contours[ID2],cv2.CONTOURS_MATCH_I1, 0)
# cv2.drawContours(imgSrc, contours, ID1, (0,0,255), 3)
# cv2.drawContours(imgSrc, contours, ID2, (0,0,255,),3)
# print('返回类型：', type(c1Andc2MatchShapeRtval))
# print('返回值：', c1Andc2MatchShapeRtval)

#以3号下标轮廓为模板，把matchShapes低于0.15值轮廓绘制出来
for i in range(len(contours)):
    rtval = cv2.matchShapes(contours[3], contours[i], cv2.CONTOURS_MATCH_I1, 0)
    if (rtval<=0.11):
        print(i, rtval)
        cv2.drawContours(imgSrc, contours, i, (0,0,255), 3)

cv2.imshow('test', imgSrc)
cv2.waitKey()

