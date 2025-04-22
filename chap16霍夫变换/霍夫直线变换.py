import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'C:\Users\11971\OneDrive\Python_project\入门\pics\轮廓.png')
img = cv2.resize(img, None, None, 0.5, 0.5)
imgHoughLine = img.copy()
imgHoughLineP = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGray = cv2.bitwise_not(imgGray)

#先进行Canny边缘检测
imgCanny = cv2.Canny(imgGray,180, 220)

#霍夫变换检测直线
lines = cv2.HoughLines(imgCanny, 1, np.pi/180, 75)
print('lines.shape: ', lines.shape)

#绘制霍夫变换检测直线的结果
for i in range(len(lines)):
    #将霍夫变换中直线的极坐标形式返回，返回形式为[rho, theta]rho为极点到直线的最短距离，theta为最短距离与x正方向夹角
    #方便cv2.line()绘制直线：cosx + siny - r = 0
    rho, theta = lines[i][0]
    costh = np.cos(theta)
    sinth = np.sin(theta)
    pt1 = None
    pt2 = None
    if np.abs(sinth) >= 1e-5:
        pt1 = (0, int((rho-costh*0)/sinth))
        pt2 = (imgGray.shape[1], int((rho-costh*imgGray.shape[1])/sinth))
    else:
        pt1 = (int((rho-sinth*0)/costh), 0)
        pt2 = (int((rho-sinth*imgGray.shape[0])/costh), imgGray.shape[0])
    cv2.line(imgHoughLine, pt1, pt2, (0,0,255), 2)


#概率霍夫变换，增加最小直线长度和直线最大两点间隔，直线总长小于最小长度或直线某两点超过最大间隔都不算做直线
#概率霍夫变换，返回值不一样，返回检测到的线段的起点和终点
linesP = cv2.HoughLinesP(imgCanny, 1, np.pi/180, 75, None, 10, 25)
print(linesP.shape)
for i in range(len(linesP)):
    x1,y1,x2,y2 = linesP[i][0]
    pt1 = (x1, y1)
    pt2 = (x2, y2)
    cv2.line(imgHoughLineP, pt1, pt2, (0,0,255), 2)


cv2.imshow('hough line', imgHoughLine)
cv2.imshow('hough line P', imgHoughLineP)
cv2.waitKey(0)
