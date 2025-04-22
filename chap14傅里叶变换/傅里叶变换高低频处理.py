import cv2
import numpy as np
import matplotlib.pyplot as plt

import chineseImgReader


#能够读取图片，实现低通滤波、高通滤波、并展示曲线
class MyFourier(object):
    def __init__(self, imgPath = None):
        self._imgPath = imgPath
        self._img = None
        self._imgGray = None
        self._imgLowPass = None
        self._imgHighPass = None
        self._fourier = None  #傅里叶转换后数据
        self._fourierShift = None  #傅里叶转换后0频分量移动至中心数据
        self._lowV = 0  #低通滤波阈值
        self._highV = 0  #高通滤波阈值
        if self._imgPath is not None:
            self.imgRead(self._imgPath)

    #读取图片（只能英文路径）
    def imgRead(self, imgPath):
        self._imgPath = self._imgPath
        self._img = cv2.imread(self._imgPath)

    #缩放
    def resizeImg(self, scale):
        self._img = cv2.resize(self._img, None, None, scale, scale)

    #傅里叶变换
    def fourierTrans(self):
        if self._img is not None:
            self._imgGray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
            self._fourier = cv2.dft(np.float32(self._imgGray), flags = cv2.DFT_COMPLEX_OUTPUT)
            self._fourierShift = np.fft.fftshift(self._fourier)
            print('计算傅里叶数据完成')

    #实现高通滤波
    def highPass(self, v):
        fourierShift = self._fourierShift.copy()
        centerRow, centerCol = self._imgGray.shape[0] // 2, self._imgGray.shape[1] // 2

        #小数百分比
        if isinstance(v, float):
            print('小数阈值')
            self._highV = int(v*(self._imgGray.shape[0]+self._imgGray.shape[1])/4)
        elif isinstance(v, int):
            print('整数阈值')
            self._highV = v
        if self._highV*2>=self._imgGray.shape[0] or self._highV*2>=self._imgGray.shape[1]:
            print('高通滤波整数阈值超出边界: >shape')
            return
        print('高通滤波归零范围：', centerRow-self._highV, centerRow+self._highV, centerCol-self._highV, centerCol+self._highV)
        fourierShift[centerRow-self._highV:centerRow+self._highV, centerCol-self._highV:centerCol+self._highV,:] = 0
        fourierIShift = np.fft.ifftshift(fourierShift)
        fourierIShift = cv2.idft(fourierIShift)
        fourierIshiftMag = cv2.magnitude(fourierIShift[:,:,0], fourierIShift[:,:,1])
        fourierIshiftMag = fourierIshiftMag/fourierIshiftMag.max()*255
        self._imgHighPass = fourierIshiftMag.astype(np.uint8)

    #实现低通滤波
    def lowPass(self, v):
        fourierShift = self._fourierShift.copy()
        centerRow, centerCol = self._imgGray.shape[0] // 2, self._imgGray.shape[1] // 2

        #小数百分比
        if isinstance(v, float):
            self._lowV = int(v*(self._imgGray.shape[0]+self._imgGray.shape[1])/4)
        elif isinstance(v, int):
            self._lowV = v

        if self._lowV<=0:
            print('低通滤波整数阈值超出边界: <=0')
            return
        if self._lowV*2>=self._imgGray.shape[0] or self._lowV*2>=self._imgGray.shape[1]:
            self._lowV = min(self._imgGray.shape)//2
        mask = np.zeros(self._imgGray.shape, dtype = np.uint8)
        mask[centerRow-self._lowV:centerRow+self._lowV, centerCol-self._lowV:centerCol+self._lowV] = 255
        fourierShift = cv2.bitwise_and(fourierShift, fourierShift, mask = mask)
        fourierIShift = np.fft.ifftshift(fourierShift)
        fourierIShift = cv2.idft(fourierIShift)
        fourierIShiftMag = cv2.magnitude(fourierIShift[:,:,0], fourierIShift[:,:,1])
        fourierIShiftMag = fourierIShiftMag/fourierIShiftMag.max()*255
        self._imgLowPass = fourierIShiftMag.astype(np.uint8)

    #绘制第一行频谱图
    def drawPlot(self):
        plt.figure('result with lowV: {} and highV: {}'.format(self._lowV, self._highV), (8,6), 100)
        if self._imgGray is not None:
            dataOrg = self._imgGray[0,:]
            plt.plot(dataOrg, label = 'org')
        if self._imgLowPass is not None:
            dataLow = self._imgLowPass[0,:]
            plt.plot(dataLow, label = 'low')
        if self._imgHighPass is not None:
            dataHigh = self._imgHighPass[0,:]
            plt.plot(dataHigh, label = 'high')
        plt.legend()


    def showImg(self):
        if self._imgGray is not None:
            cv2.imshow('imgGray', self._imgGray)
        if self._imgLowPass is not None:
            cv2.imshow('imgLowPass: {}'.format(self._lowV), self._imgLowPass)
        if self._imgHighPass is not None:
            cv2.imshow('imgHighPass: {}'.format(self._highV), self._imgHighPass)
        plt.show()
        cv2.waitKey(0)


if __name__ == '__main__':
    myFourier = MyFourier(r'./pics/poker.png')
    # myFourier.resizeImg(0.4)
    #进行傅里叶变换
    myFourier.fourierTrans()
    #进行高通滤波
    v = 0.02
    myFourier.lowPass(v)
    myFourier.highPass(v)
    myFourier.drawPlot()
    myFourier.showImg()