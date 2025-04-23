import cv2
import numpy as np

import chineseImgReader

#读取原始图像
img = chineseImgReader.imgRead(r'C:\Users\LFK\OneDrive\Python_project\learning\pics\coins.jpg')
scaleSize = 0.5
img = cv2.resize(img, None, None, scaleSize, scaleSize)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# img[:,:,2] = cv2.equalizeHist(img[:,:,2])
# img[:,:,1] = cv2.equalizeHist(img[:,:,1])
# img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

print('img info: ', img.shape, img.dtype, img.min(), img.max())
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGray = cv2.GaussianBlur(imgGray, (7,7), 0)

#阈值分割
# t, imgThresh = cv2.threshold(imgGray, 120, 255, cv2.THRESH_BINARY)
t, imgThresh = cv2.threshold(imgGray, 90, 255, cv2.THRESH_BINARY_INV)

#形态学开运算
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
imgThresh = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, k, None, (-1,-1), 1)

#距离变换
imgDistance = cv2.distanceTransform(imgThresh, cv2.DIST_L2, 3)
print('img distance info: ', imgDistance.shape, imgDistance.dtype, imgDistance.min(), imgDistance.max())

#距离变换阈值处理，提取前景，前景是白色255
t, imgFront = cv2.threshold(imgDistance, 0.8*imgDistance.max(), 255, cv2.THRESH_BINARY)
imgFront = np.uint8(imgFront)
print('imgFront info: ', imgFront.shape, imgFront.dtype, imgFront.min(), imgFront.max())

#将原图阈值膨胀确定背景（背景就是黑色）
imgBack = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, k, None, (-1,-1), 1)

#计算未知区域，或者叫不确定区域，白色255
imgUn = cv2.subtract(imgBack, imgFront)  #减法，背景-前景
cv2.imshow('imgFront: ', imgFront)
cv2.imshow('imgBack: ', imgBack)
cv2.imshow('imgUn: ', imgUn)
cv2.waitKey(0)

#在前景上打标记
markCnt, imgMark = cv2.connectedComponents(imgFront)  #标签个数+标签掩码
print('imgMark info: ', imgMark.shape, imgMark.dtype, imgMark.min(), imgMark.max(), markCnt)

#将标签变为分水岭算法需要的形式，分水岭算法需要规定未知区域为0
imgMark += 1  #确保没有0
imgMark[imgUn==255] = 0  #将未知区域变为0

#使用分水岭算法
cv2.watershed(img, imgMark)  #修改实参imgMark，-1为边缘（包括图像四周）
print('imgMark after watershed info: ', imgMark.shape, imgMark.dtype, imgMark.min(), imgMark.max())

#将分水岭的边缘展示在原图中
img[imgMark==-1] = (0,0,255)

#随机颜色展示效果
# imgResult = np.zeros(img.shape, dtype = np.uint8)
imgResult = img.copy()
for i in range(2, markCnt+1):
    colorRandom = (np.random.randint(0,256),
                   np.random.randint(0,256),
                   np.random.randint(0,256))
    imgResult[imgMark==i] = colorRandom
imgResult[imgMark==-1] = (0,0,255)
imgResult[imgMark==1] = (255,255,255)

# cv2.imshow('imgThresh', imgThresh)
# cv2.imshow('img front', imgFront)
# cv2.imshow('img back', imgBack)
# cv2.imshow('img Unknow', imgUn)
cv2.imshow('img after water', img)
cv2.imshow('imgResult with random color', imgResult)
cv2.waitKey(0)