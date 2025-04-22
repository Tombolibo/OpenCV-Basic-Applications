import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/轮廓2五角星七角星.png')
img = cv2.resize(img, None, None, .4, .4)
imgCopy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshT, img = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#对每个轮廓进行多边形逼近
contoursPoly = []
for i in range(len(contours)):
    contourArcLength = cv2.arcLength(contours[i], True)
    eps = 0.01*contourArcLength
    poly = cv2.approxPolyDP(contours[i], eps, True)
    contoursPoly.append(poly)

#计算每个多边形逼近轮廓的凸包（凸包返回具体点）
hulls = []
for i in range(len(contours)):
    hulls.append(cv2.convexHull(contoursPoly[i]))

hullsIndexofContour = []
for i in range(len(contours)):
    hullsIndexofContour.append(cv2.convexHull(contoursPoly[i], returnPoints=False))

#绘制每个凸包
for i in range(len(contours)):
    cv2.drawContours(imgCopy, hulls, i, (0,0,0), 2)


#计算逼近多边形轮廓的每个凸缺陷
convexityDefectsofEachPolyContour = []
for i in range(len(contours)):
    convexityDefectsofEachPolyContour.append(cv2.convexityDefects(contoursPoly[i], hullsIndexofContour[i]))

#绘制每个凸缺陷
for i in range(len(contours)):
    for j in range(len(convexityDefectsofEachPolyContour[i])):
        farestPoint = contoursPoly[i][convexityDefectsofEachPolyContour[i][j][0][2]][0]
        #将最远点打在图像上
        cv2.circle(imgCopy, farestPoint, 3, (0,0,0), 2)


cv2.imshow('convexHull and convexity defects', imgCopy)
cv2.waitKey(0)

#检查每个轮廓是否是凸的
ifContourConvex = [cv2.isContourConvex(contoursPoly[i]) for i in range(len(contours))]
print(ifContourConvex)

#计算点和多边形、轮廓、凸包等点集的距离或关系（上、内、外）
testPoint = (100,100)
#计算这个点在多边形逼近轮廓中的关系
print(cv2.pointPolygonTest(contoursPoly[0], testPoint, True))