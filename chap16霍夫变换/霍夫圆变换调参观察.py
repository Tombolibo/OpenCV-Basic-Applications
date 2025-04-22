import cv2
import numpy as np

import chineseImgReader


class MyHoughCircle(object):
    def __init__(self):
        self._img = None
        self._imgGray = None
        self._imgCanny = None
        self._imgHoughCircle = None
        self._minDist = None
        self._param1 = None
        self._param2 = None
        self._minRadius = None
        self._maxRadius = None
        self._windowName = None
        self._keyCode = None

    #读取图片并预处理
    def imgRead(self, imgPath, ifScale = False, scale = 1.0):
        self._img = chineseImgReader.imgRead(imgPath)
        if ifScale:
            self._img = cv2.resize(self._img, None, None, scale, scale)
        self._imgGray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        self._imgGray = cv2.GaussianBlur(self._imgGray, (5, 5), 0)

    #每个属性的set方法
    def setMinDist(self, minDist):
        self._minDist = minDist
        self.hough()
    def setParam1(self, param1):
        self._param1 = param1
        self.hough()
    def setParam2(self, param2):
        self._param2 = param2
        self.hough()
    def setMinRadius(self, minRadius):
        self._minRadius = minRadius
        self.hough()
    def setMaxRadius(self, maxRadius):
        self._maxRadius = maxRadius
        self.hough()

    #窗口设置
    def createWindow(self, windowName):
        self._windowName = windowName
        cv2.namedWindow(windowName)
        cv2.namedWindow('aditional data')

    #创建滑动条
    def createTrackBars(self):
        cv2.createTrackbar('minDist', self._windowName, 20, 250, self.setMinDist)
        cv2.setTrackbarMin('minDist', self._windowName, 1)
        cv2.setTrackbarMax('minDist', self._windowName, 250)

        cv2.createTrackbar('param1', self._windowName, 1, 300, self.setParam1)
        cv2.setTrackbarMin('param1', self._windowName, 1)
        cv2.setTrackbarMax('param1', self._windowName, 300)

        cv2.createTrackbar('param2', self._windowName, 50, 150, self.setParam2)
        cv2.setTrackbarMin('param2', self._windowName, 1)
        cv2.setTrackbarMax('param2', self._windowName, 150)

        cv2.createTrackbar('minRaaius', self._windowName, 10,200, self.setMinRadius)
        cv2.setTrackbarMin('minRaaius', self._windowName, 0)
        cv2.setTrackbarMax('minRaaius', self._windowName, 200)

        cv2.createTrackbar('maxRadius', self._windowName, 20,300, self.setMaxRadius)
        cv2.setTrackbarMin('maxRadius', self._windowName, 0)
        cv2.setTrackbarMax('maxRadius', self._windowName, 300)
        print('滑动条完成')


    #计算霍夫变换
    def hough(self):
        if self._minDist is not None and\
            self._param1 is not None and\
            self._param2 is not None and\
            self._minRadius is not None and\
            self._maxRadius is not None:
            circles = cv2.HoughCircles(self._imgGray, cv2.HOUGH_GRADIENT, 1, self._minDist,
                                       None, self._param1, self._param2, self._minRadius, self._maxRadius)
            if circles is not None:
                print('circle.shape: ', circles.shape)
                self._imgHoughCircle = self._img.copy()
                for i in range(len(circles[0])):
                    x,y,r = circles[0][i]
                    x = int(x)
                    y = int(y)
                    r = int(r)
                    print(x,y,r)
                    cv2.circle(self._imgHoughCircle, (x,y), r, (0,0,255), 2)
            else:
                print('无圆')
                self._imgHoughCircle = self._img.copy()
        else:
            print('参数不完整')
            self._imgHoughCircle = self._img.copy()

    #展示图片
    def imgShow(self):
        if self._imgHoughCircle is not None:
            self._imgCanny = cv2.Canny(self._imgGray, self._param1//2, self._param1)
            img = np.concatenate([self._imgGray, self._imgCanny], axis=1)
            cv2.imshow(self._windowName, self._imgHoughCircle)
            cv2.imshow('aditional data', img)
        self._keyCode = cv2.waitKey(30)


    #结束，销毁窗口
    def destroy(self):
        cv2.destroyAllWindows()

    #运行
    def run(self, imgPath, windowName, ifScale=False, scale=1.0):
        #读取图片
        self.imgRead(imgPath, ifScale, scale)
        #创建窗口
        self.createWindow(windowName)
        #创建滑动条
        self.createTrackBars()
        #展示图像
        while self._keyCode != 27:
            self.imgShow()
        print('结束')

if __name__ == '__main__':
    myHough = MyHoughCircle()
    myHough.run(r'C:\Users\LFK\OneDrive\Python_project\learning\pics\coins.jpg', 'test', True, 0.3)