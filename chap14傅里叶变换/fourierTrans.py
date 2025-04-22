import cv2
import numpy as np

import chineseImgReader

#通过调整fourierCenterData位置数据，改变值

class FourierTrans(object):
    def __init__(self):
        self._imgSrc = None
        self._imgShow = None
        self._imgFourierCenterShow = None
        self._imgPath = None
        self._windowName = None
        self._windowExisted = False
        self._fourierData = None
        self._fourierDataCenter = None
        self._fourierDataCenterMagnitude = None
        self._fourierDataCopy = None
        self._iFourierData = None
        self._iFourierDataCenter = None
        self._iFourierDataCenterMagnitude = None
        self._changex = 0
        self._changey = 0

    #读取图片
    def imgRead(self, imgPath):
        print('读取图片')
        if imgPath is not None:
            self._imgPath = imgPath;
            self._imgSrc = chineseImgReader.imgRead(self._imgPath)
            self._imgSrc = cv2.cvtColor(self._imgSrc, cv2.COLOR_BGR2GRAY)
            self._imgSrc = cv2.equalizeHist(self._imgSrc)

    #创建窗口
    def createWondow(self, windowName):
        if windowName is not None:
            self._windowName = windowName
            cv2.namedWindow(self._windowName)
            self._windowExisted = True
    def destroyWindow(self):
        cv2.destroyAllWindows()
        self._windowExisted = False

    #创建滑动条
    def createTrackBar(self, barName, maxV, func):
        if self._imgSrc is not None:
            cv2.createTrackbar(barName, self._windowName, 0,maxV, func)

    #进行傅里叶变换
    def fourierTrans(self):
        print('傅里叶转换')
        print(self._imgSrc.shape)
        if self._imgSrc is not None:
            print('转换开始')
            self._fourierData = cv2.dft(np.float32(self._imgSrc), flags=cv2.DFT_COMPLEX_OUTPUT)
            print(self._fourierData.shape)
            self._fourierDataCenter = np.fft.fftshift(self._fourierData)
            self._fourierDataCenterMagnitude = cv2.magnitude(self._fourierDataCenter[:,:,0],
                                                             self._fourierDataCenter[:,:,1])
            # self._imgFourierCenterShow = self._fourierDataCenterMagnitude/np.max(self._fourierDataCenterMagnitude)*255
            self._imgFourierCenterShow = 20*np.log(self._fourierDataCenterMagnitude)
            self._imgFourierCenterShow = self._imgFourierCenterShow.astype(np.uint8)
            print('傅里叶转换完成：', self._imgFourierCenterShow.shape)
        else:
            print('图片为空')
    def fourierCopy(self):
        if self._fourierData is not None:
            self._fourierDataCopy = self._fourierData.copy()

    def iFourierTrans(self):
        if (self._fourierDataCopy is not None):
            self._iFourierData = cv2.idft(self._fourierDataCopy)
            self._iFourierDataCenterMagnitude = cv2.magnitude(self._iFourierData[:,:,0], self._iFourierData[:,:,1])
            self._imgShow = self._iFourierDataCenterMagnitude/np.max(self._iFourierDataCenterMagnitude)*255
            self._imgShow = self._imgShow.astype(np.uint8)

    def setChangex(self, pos):
        self._changex = pos
    def setChangey(self, pos):
        self._changey = pos
    def changeBlack(self, x, y):
        #将傅里叶变换后以中间为圆心，2x，2y为边长的正方形区域变黑
        self._fourierDataCopy = np.fft.fftshift(self._fourierDataCopy)
        shape = self._fourierDataCopy.shape
        self._fourierDataCopy[shape[0]//2-y:shape[0]//2+y, shape[1]//2-x:shape[1]//2+x, :] = 0
        magnitudeTemp = cv2.magnitude(self._fourierDataCopy[:,:,0], self._fourierDataCopy[:,:,1])
        magnitudeTemp = 20*np.log(magnitudeTemp)
        self._imgFourierCenterShow = magnitudeTemp.astype(np.uint8)
        self._fourierDataCopy = np.fft.ifftshift(self._fourierDataCopy)


    def setUp(self, imgPath, imgWindowName, fourierWindowName):
        self.imgRead(imgPath)
        self.createWondow(imgWindowName)
        self.createWondow(fourierWindowName)
        self.createTrackBar('x', self._imgSrc.shape[1]//2, self.setChangex)
        self.createTrackBar('y', self._imgSrc.shape[0]//2,self.setChangey)

    def run(self, imgPath, imgWindowName, fourierWindowName):
        self.setUp(imgPath, imgWindowName, fourierWindowName)
        #进行傅里叶变换，只进行一次
        self.fourierTrans()
        while(self._windowExisted):
            cv2.imshow(fourierWindowName, self._imgFourierCenterShow)

            #修改频域
            self.fourierCopy()
            self.changeBlack(self._changex, self._changey)

            #每次，进行傅里叶逆变换
            self.iFourierTrans()
            cv2.imshow(imgWindowName, self._imgShow)
            keyCode = cv2.waitKey(1)
            if (keyCode == ord('b')):
                self.destroyWindow()


if __name__ == '__main__':
    myFourier = FourierTrans()
    myFourier.run(r'./pics/斯卡利茨.png', 'i img', 'fourier')




