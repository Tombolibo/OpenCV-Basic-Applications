import numpy as np
import cv2
import matplotlib.pyplot as plt

import chineseImgReader

# imgSrc = chineseImgReader.imgRead(r'./pics/斯卡利茨.png')
imgSrc = cv2.imread(r'./pics/rand2.bmp')
imgGray = cv2.cvtColor(imgSrc, cv2.COLOR_BGRA2GRAY)
#直方图均衡
imgGray = cv2.equalizeHist(imgGray)
rows,cols = imgGray.shape
print('imgGray.shape: ', imgGray.shape)

#把原始灰度图中间行的变化缓存
imgGrayMidLineData = imgGray[rows//2,:]

#傅里叶变换
imgFourier = cv2.dft(np.float32(imgGray), flags=cv2.DFT_COMPLEX_OUTPUT)  #注意数据类型转换以及输出格式(flags = )
imgFourier = np.fft.fftshift(imgFourier)
print('imgFourier.shape: ', imgFourier.shape)

#高通滤波（+-rangeW）掩膜
# rangeH = np.min([rows,cols])//8
rangeH = 20
print('rangeH: ', rangeH)
highPassMask = np.ones((rows, cols, 2),np.uint8)
highPassMask[rows//2-rangeH:rows//2+rangeH, cols//2-rangeH:cols//2+rangeH,:] = 0
fourierHighPass = imgFourier*highPassMask

#低通滤波掩膜
lowPassMask = highPassMask.copy()
lowPassMask[highPassMask==1] = 0
lowPassMask[highPassMask==0] = 1
fourierLowPass = imgFourier*lowPassMask

#对高通滤波和低通滤波频域操作结果进行逆傅里叶变换
#高通滤波后逆变换
iFourierHighPass = np.fft.ifftshift(fourierHighPass)
iFourierHighPass = cv2.idft(iFourierHighPass)
iFourierHighPassImg = cv2.magnitude(iFourierHighPass[:,:,0], iFourierHighPass[:,:,1])
iFourierHighPassImg = iFourierHighPassImg/np.max(iFourierHighPassImg)*255
iFourierHighPassImg = iFourierHighPassImg.astype(np.uint8)

#低通滤波后逆变换
iFourierLowPass = np.fft.ifftshift(fourierLowPass)
iFourierLowPass = cv2.idft(iFourierLowPass)
iFourierLowPassImg = cv2.magnitude(iFourierLowPass[:,:,0], iFourierLowPass[:,:,1])
print('幅值类型：', iFourierLowPassImg.dtype)
# iFourierLowPassImg = iFourierLowPassImg/np.max(iFourierLowPassImg)*255
iFourierLowPassImg = cv2.normalize(iFourierLowPassImg, None, 0,255, cv2.NORM_MINMAX)
print('min max slahe', iFourierLowPass.shape)
iFourierLowPassImg = iFourierLowPassImg.astype(np.uint8)


#显示图片
cv2.imshow('imgSrc', imgGray)
cv2.imshow('highPass', iFourierHighPassImg)
cv2.imshow('lowPass', iFourierLowPassImg)

cv2.waitKey()

#显示中间行的波
imgLowPassMidLineData = iFourierLowPassImg[rows//2,:]
imgHighPassMidLineData = iFourierHighPassImg[rows//2,:]

plt.plot(imgGrayMidLineData, label = 'imgGray')
plt.plot(imgLowPassMidLineData, label = 'low pass')
plt.plot(imgHighPassMidLineData, label = 'high pass')
plt.legend()
plt.show()


