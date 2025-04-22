import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/轮廓2五角星七角星.png')
img = cv2.resize(img, None, None, 0.4, 0.4)
imgCopy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshT, img = cv2.threshold(img, -1,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#检测外轮廓
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


#在源图像上绘制矩形包围框
imgRect = imgCopy.copy()
for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])
    cv2.rectangle(imgRect, [x,y], [x+w, y+h], (0,0,0),3)
# cv2.imshow('imgRectCopy', imgRect)
# cv2.waitKey(0)

#最小矩形包围框
imgMinRect = imgCopy.copy()
for i in range(len(contours)):
    minRectInfo = cv2.minAreaRect(contours[i]) #返回类型为((矩形中心x， 矩形中心y),(宽度，高度),旋转角度)
    #可以使用cv2.boxPoints()转换为矩形轮廓
    minRectContour = cv2.boxPoints(minRectInfo)
    minRectContour = np.array(minRectContour, dtype=np.int32)  # cv2.drawContours只接受整数轮廓参数
    cv2.drawContours(imgMinRect, [minRectContour] , 0, (0,0,0), 2)
# cv2.imshow('imgMinRect', imgMinRect)
# cv2.waitKey(0)

#最小外包圆形
imgMinCircle = imgCopy.copy()
for i in range(len(contours)):
    center, r = cv2.minEnclosingCircle(contours[i])
    center = np.array(center, dtype = np.int32)
    r = int(r)
    cv2.circle(imgMinCircle, center, r, (0,0,0), 3)
# cv2.imshow('min enclosing circle', imgMinCircle)
# cv2.waitKey(0)

#最优拟合椭圆
imgMinEllipse = imgCopy.copy()
for i in range(len(contours)):
    ret = cv2.fitEllipse(contours[i])
    cv2.ellipse(imgMinEllipse, ret, (0,0,0),3)

    ellipseRectPoints = cv2.boxPoints(ret)
    ellipseRectPoints = np.array(ellipseRectPoints, dtype = np.int32)
    cv2.drawContours(imgMinEllipse, [ellipseRectPoints],0,(0,0,0),2)
# cv2.imshow('min ellipse', imgMinEllipse)
# cv2.waitKey(0)

#最优拟合直线
imgMinLine = imgCopy.copy()
for i in range(len(contours)):
    [vx, vy, x1, y1] = cv2.fitLine(contours[i], cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    k = vy/vx
    b = y1 - k*x1
    pt1 = (0, int(b))
    pt2 = (imgMinLine.shape[1], int(k*imgMinLine.shape[1]+b))
    cv2.line(imgMinLine, pt1, pt2, (0,0,0), 2)
# cv2.imshow('min line', imgMinLine)
# cv2.waitKey(0)


#最优拟合三角形
imgMinTriangle = imgCopy.copy()
for i in range(len(contours)):
    triangleArea, trianglePoints = cv2.minEnclosingTriangle(contours[i])
    trianglePoints = np.array(trianglePoints, dtype = np.int32)
    cv2.line(imgMinTriangle, trianglePoints[0][0], trianglePoints[1][0], (0,0,0), 2)
    cv2.line(imgMinTriangle, trianglePoints[0][0], trianglePoints[2][0], (0,0,0), 2)
    cv2.line(imgMinTriangle, trianglePoints[1][0], trianglePoints[2][0], (0,0,0), 2)
# cv2.imshow('test', imgMinTriangle)
# cv2.waitKey(0)