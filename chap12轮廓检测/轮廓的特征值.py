import numpy as np
import cv2

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/轮廓2五角星七角星.png')
img = cv2.resize(img, None, None, 0.4, 0.4)
imgCopy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshT, img = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#轮廓识别
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#轮廓多边形近似
contoursPoly = [cv2.approxPolyDP(contours[i], 0.01*cv2.arcLength(contours[i], True), True) for i in range(len(contours))]

#轮廓高宽比
#计算外界矩阵
for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contoursPoly[i])
    print('轮廓{}高宽比:'.format(i), w/h)
print('----------------------------------------------------------------')
#轮廓extend（轮廓面积/矩形面积）
for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contoursPoly[i])
    print('轮廓{}extend：'.format(i), cv2.contourArea(contoursPoly[i])/(w*h))

#solidity（轮廓面积/凸包面积）
convexHull = [cv2.convexHull(contoursPoly[i]) for i in range(len(contours))]
for i in range(len(contours)):
    print('轮廓{} solidity：'.format(i), cv2.contourArea(contoursPoly[i])/cv2.contourArea(convexHull[i]))

#等效直径（与轮廓面积相等圆的直径）
for i in range(len(contours)):
    print('轮廓{}等效直径：'.format(i), np.sqrt(4*cv2.contourArea(contoursPoly[i])/np.pi))


#极点
for i in range(len(contours)):
    print('左：', contoursPoly[i][contoursPoly[i][:,:,0].argmin()][0])
    print('右：', contoursPoly[i][contoursPoly[i][:,:,0].argmax()][0])
    print('上：', contoursPoly[i][contoursPoly[i][:,:,1].argmin()][0])
    print('下：', contoursPoly[i][contoursPoly[i][:,:,1].argmax()][0])
