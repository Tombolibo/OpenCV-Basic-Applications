import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/斯卡利茨.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGray = np.concatenate([imgGray, imgGray], axis = 1)
imgGray = np.concatenate([imgGray, imgGray], axis = 0)
print(imgGray.shape)
tempLate = imgGray[50:100, 300:400].copy()
print('template.shape: ', tempLate.shape)

#进行多模板匹配
templateResult = cv2.matchTemplate(imgGray, tempLate, cv2.TM_SQDIFF_NORMED)  #标准差
print('match result.shape: ', templateResult.shape)
#查找多模板匹配中满足阈值的位置
myThresh = 0.003
points = np.where(templateResult<myThresh)
print('查找到多模板个数：', len(points[0]))
#绘制矩形
tempH = tempLate.shape[0]
tempW = tempLate.shape[1]
for i in zip(*points):
    cv2.rectangle(imgGray, (i[1],i[0]), (i[1]+tempW, i[0]+tempH), 255, 1)



cv2.imshow('test', imgGray)
cv2.imshow('template', tempLate)
cv2.waitKey(0)