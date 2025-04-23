import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'C:\Users\LFK\OneDrive\Python_project\learning\pics\person.jpg')
print('img info: ', img.shape)

#掩膜、背景参数、前景参数、矩形
mask = np.zeros(img.shape[:2], dtype = np.uint8)  # 背景0
mask[75:225, 250:400] = 2  #可能背景
mask[100:200, 300:350] = 3  #可能前景
mask[110:180, 310:320] = 1  #前景
maskShow = mask.copy()
maskShow = maskShow*85

maskAfter = mask.copy()
# grabcut需要的两个向量
bgdModel = np.zeros((1,65), dtype = np.float64)
fgdModel = np.zeros((1,65), dtype = np.float64)
rect = (220, 50, 200,200)
# rect = (220, 50, 10,10)
cv2.rectangle(img, rect[:2], (rect[0]+rect[2], rect[1]+rect[3]), (0,0,255),3)

#进行使用初始掩膜的提取
cv2.grabCut(img, maskAfter, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

#将提取信息进行处理
maskAfterShow = maskAfter.copy()
maskAfterShow = maskAfterShow*85
maskAfterShow = maskAfterShow.astype(np.uint8)
maskAfter[(maskAfter==0) | (maskAfter==2)] = 0
maskAfter[(maskAfter==1) | (maskAfter==3)] = 1
maskAfter = maskAfter*255
maskAfter = maskAfter.astype(np.uint8)

#提取图片
imgCut = cv2.bitwise_and(img, img, mask = maskAfter)

#展示提取图片
cv2.imshow('img src', img)
cv2.imshow('mask', maskShow)
cv2.imshow('mask after', maskAfterShow)
cv2.imshow('img cut', imgCut)
print('bgdModel: ', bgdModel)
print('fgdModel: ', fgdModel)

cv2.waitKey(0)
