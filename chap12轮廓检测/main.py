import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/轮廓.png')
img = cv2.resize(img, None, None, 0.5, 0.5)
imgCopy = img.copy()

#将图像转换为二值图
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#查找轮廓
contours_of_img, hierarchy_of_img = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


#展示信息
print('轮廓个数：', len(contours_of_img))
print('contours[0].shape', contours_of_img[0].shape)
for i in range(len(contours_of_img)):
    print(i, hierarchy_of_img[0][i])


#给每个轮廓打上标签
for i in range(len(contours_of_img)):
    cv2.putText(imgCopy, str(i), contours_of_img[i][0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)

#绘制轮廓
for i in range(len(contours_of_img)):
    cv2.drawContours(imgCopy, contours_of_img, i, (0,0,255), 2)


#计算每个图像的特征矩
moments = []
hus = []
for i in range(len(contours_of_img)):
    moment = cv2.moments(contours_of_img[i], False)
    #Hu矩：归一化中心矩的线性组合
    hu = cv2.HuMoments(moment)
    moments.append(moment)
    hus.append(hu)
    print(moment, hu, sep = '\n', end = '\n-----------------------------\n')

#利用cv2的matchShapes()进行轮廓或者灰度图像的hu矩形状匹配
print('cv2的matchShapes()进行轮廓或者灰度图像的hu矩形状匹配')
for i in range(len(contours_of_img)):
    for j in range(len(contours_of_img)):
        print(i,j,cv2.matchShapes(contours_of_img[i], contours_of_img[j], cv2.CONTOURS_MATCH_I3, 0))
        #返回值是差异信息，完全相同轮廓的差异为0

#轮廓的常用信息
for i in range(len(contours_of_img)):
    print('----------------', i, '----------------')
    #面积
    print('面积：', cv2.contourArea(contours_of_img[i], False))
    #轮廓长度
    print('轮廓长度：', cv2.arcLength(contours_of_img[i], True))
    #轮廓面积和m00值相同
