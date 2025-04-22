import cv2
import numpy as np

import chineseImgReader



imgSrc = chineseImgReader.imgRead(r'./pics/轮廓2五角星七角星.png')
imgSrc = cv2.resize(imgSrc, (int(imgSrc.shape[1]*0.3), int(imgSrc.shape[0]*0.3)))
imgGray = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)
_, imgThresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)

#获取轮廓
contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)




#获取每个多边形逼近轮廓的凸包
imgSrcBack1 = imgSrc.copy()
hulls = []
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i], returnPoints=False)
    hulls.append(hull)

#针对获得的某个凸包计算其，凸缺陷
hullsDefects = []
for i in range(len(hulls)):
    defect = cv2.convexityDefects(contours[i], hulls[i])
    hullsDefects.append(defect)


#注意现在使用的轮廓是逼近了多边形的轮廓
# cv2.line(imgSrcBack1, ploylist[0][hullsDefects[0][0][0][0]][0], ploylist[0][hullsDefects[0][0][0][1]][0], (0,0,0),2)
# cv2.line(imgSrcBack1, (ploylist[0][hullsDefects[0][0][0][0]][0]+ploylist[0][hullsDefects[0][0][0][1]][0])//2, ploylist[0][hullsDefects[0][0][0][2]][0], (0,0,255),2)

#把每个轮廓的每个凸缺陷绘制出来
for i in range(len(contours)):
    #每个凸缺陷
    print(str(i)+'号', len(hullsDefects[i]),'角星')
    for j in range(len(hullsDefects[i])):
        startPoint = contours[i][hullsDefects[i][j][0][0]][0]
        endPoint = contours[i][hullsDefects[i][j][0][1]][0]
        farPoint = contours[i][hullsDefects[i][j][0][2]][0]
        cv2.line(imgSrcBack1, startPoint, endPoint, (0,0,0),2)
        cv2.line(imgSrcBack1, (startPoint+endPoint)//2, farPoint, (255,0,0),2)


cv2.imshow('test', imgSrcBack1)
cv2.waitKey()