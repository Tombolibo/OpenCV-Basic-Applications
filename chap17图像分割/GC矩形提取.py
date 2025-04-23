import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'C:\Users\LFK\OneDrive\Python_project\learning\pics\person.jpg')
print('img info: ', img.shape)

#掩膜、背景参数、前景参数、矩形
mask = np.zeros(img.shape[:2], dtype = np.uint8)
maskAfter = mask.copy()
bgdModel = np.zeros((1,65), dtype = np.float64)
fgdModel = np.zeros((1,65), dtype = np.float64)
# 矩形位置
rect = (220, 50, 200,200)
cv2.rectangle(img, rect[:2], (rect[0]+rect[2], rect[1]+rect[3]), (255,255,255),3)

#进行只使用矩形的交互式提取
cv2.grabCut(img, maskAfter, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

#将提取信息进行处理
maskAfterShow = maskAfter*85
maskAfter[(maskAfter==0) | (maskAfter==2)] = 0
maskAfter[(maskAfter==1) | (maskAfter==3)] = 1
maskAfterShow = maskAfterShow.astype(np.uint8)

#提取图片
imgCut = cv2.bitwise_and(img, img, mask = maskAfter)

#展示提取图片
cv2.imshow('img src', img)
cv2.imshow('mask', mask)
cv2.imshow('mask after', maskAfterShow)
cv2.imshow('img cut', imgCut)
print('bgdModel: ', bgdModel)
print('fgdModel: ', fgdModel)

cv2.waitKey(0)
