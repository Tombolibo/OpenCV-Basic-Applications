import cv2
import numpy as np
import matplotlib.pyplot as plt

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/斯卡利茨.png')
img = cv2.resize(img, None, None, 1.2, 1.2)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGray = cv2.equalizeHist(imgGray)
print('imgGray.shape: ', imgGray.shape)

#使用numpy进行傅里叶变换
fourierData = np.fft.fft2(imgGray)  #0频分量在左上
fourierDataShift = np.fft.fftshift(fourierData)  #将0频分量移至中间，周围是高频分量
fourierDataShiftImg = 20*np.log(np.abs(fourierDataShift))  #计算频域幅度并映射到[0,255]，忽略相位（图像处理大多用幅度）
fourierDataShiftImg = fourierDataShiftImg.astype(np.uint8)  #注意数据类型转换

#numpy实现逆傅里叶变换
fourierDataShift2Org = np.fft.ifftshift(fourierDataShift)  #将转换后0频分量在中间的频谱移至0频分量在左上角的频谱
imgGrayIFourier = np.fft.ifft2(fourierDataShift2Org)  #逆变换后也是复数数组，要映射到[0,255]
imgGrayIFourier = np.abs(imgGrayIFourier)
imgGrayIFourier = imgGrayIFourier.astype(np.uint8)

cv2.imshow('imgsrc', imgGray)
cv2.imshow('fourier trans of img', fourierDataShiftImg)
cv2.imshow('img gray iFourier', imgGrayIFourier)
cv2.waitKey(0)


