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
myThresh = 0.01  # 模板匹配的阈值
points = np.where(templateResult<myThresh)  # 返回一个元组
print('查找到多模板个数：', len(points[0]))

#绘制与保存矩形信息（矩形位置和置信度）
tempH = tempLate.shape[0]
tempW = tempLate.shape[1]
rects = []
rectsConf = []
for i in zip(*points):
    rects.append([i[1], i[0], i[1]+tempW, i[0]+tempH])
    rectsConf.append(1-templateResult[i[0], i[1]])

# 对查找到的模板进行非极大值抑制（需要矩形列表以及每个矩形对应的置信度列表）
NMSresult = cv2.dnn.NMSBoxes(bboxes=rects, scores=rectsConf, score_threshold=1-myThresh, nms_threshold = 0.5)
print('NMSresult', NMSresult)
print('非极大值抑制后矩形个数：', len(NMSresult))

# 绘制非极大值抑制后的矩形结果
for index in NMSresult:
    cv2.rectangle(imgGray, rects[index][:2], rects[index][2:], (255,255,255), 3)



cv2.imshow('test', imgGray)
cv2.imshow('template', tempLate)
cv2.waitKey(0)