import numpy as np
import cv2

import chineseImgReader

img = cv2.imread(r'./pics/rand2.bmp')
img = cv2.resize(img, None, None, 0.5, 0.5)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('imgGray.shape: ', imgGray.shape)

#opencv进行傅里叶变换
fourierData = cv2.dft(np.float32(imgGray), None, flags=cv2.DFT_COMPLEX_OUTPUT)  #输出复数阵列，不过是2个通道，0为实部、1为虚部
fourierDataShift = np.fft.fftshift(fourierData)  #将零频分量移至中心
fourierDataShiftMagnitude = cv2.magnitude(fourierDataShift[:,:,0], fourierDataShift[:,:,1])  #计算幅度
# fourierImg = 20*np.log(fourierDataShiftMagnitude)
fourierImg = fourierDataShiftMagnitude/fourierDataShiftMagnitude.max()*255
fourierImg = fourierImg.astype(np.uint8)
#傅里叶逆变换
fourierIShift = np.fft.ifftshift(fourierDataShift)  #将0频分量从中间转换到左上角
fourierIShiftImg = cv2.idft(fourierIShift)
fourierIShiftImg = cv2.magnitude(fourierIShiftImg[:,:,0], fourierIShiftImg[:,:,1])
fourierIShiftImg = fourierIShiftImg/fourierIShiftImg.max()*255
fourierIShiftImg = fourierIShiftImg.astype(np.uint8)
cv2.imshow('img src', imgGray)
cv2.imshow('img fourier', fourierImg)
cv2.imshow('img fourier ishift', fourierIShiftImg)
cv2.waitKey(0)