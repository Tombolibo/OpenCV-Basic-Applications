import os

import cv2
import numpy as np

class Rotate360(object):
    def __init__(self, imgPath, ifScale = False, newSize = 1):
        self._imgPath = imgPath
        self._ifScale = ifScale
        self._newSize = newSize
        self._img = None
        self._imgHeight = None
        self._imgWidth = None
        self._imgRotated = None
        self._imgRotatedHeight = None
        self._imgRotatedWidth = None
        self._rotateAngel = 0
        self._rotateAngelRad = 0
        self._M = None
        self._rotationCenter = (0,0)
        self._windowName = None
        self._windowExisted = False
        self.readImg(self._imgPath)

    #读取图片
    def readImg(self, imgPath):
        self._img = cv2.imread(imgPath)
        if self._ifScale:
            self.resizeImg(self._newSize)
        self._imgHeight = self._img.shape[0]
        self._imgWidth = self._img.shape[1]

    #缩放图片
    def resizeImg(self, newSize):
        if self._img is not None and self._ifScale:
            self._img = cv2.resize(self._img, None, None, self._newSize, self._newSize)

    #增加旋转度数
    def increaseRotateAngel(self, addAngel):
        self._rotateAngel+=addAngel
        self._rotateAngel%=360
        self._rotateAngelRad = np.deg2rad(self._rotateAngel)
    #设置旋转读度数
    def setRotateAngel(self, angel = 0):
        self._rotateAngel = angel
        self._rotateAngelRad = np.deg2rad(self._rotateAngel)
    #设置旋转中心
    def setRotationCenter(self, center = (0,0)):
        self._rotationCenter = center
    #计算旋转所要的数据
    def calculateRotationData(self):
        if self._img is not None:
            M = cv2.getRotationMatrix2D(self._rotationCenter, self._rotateAngel,1)  #旋转中心，角度(deg)，缩放（不需要原始图片）
            #计算旋转后，能够完全显示图片的高和宽
            self._imgRotatedHeight = int(self._imgHeight*np.abs(np.cos(self._rotateAngelRad))+self._imgWidth*np.abs(np.sin(self._rotateAngelRad)))
            self._imgRotatedWidth = int(self._imgHeight*np.abs(np.sin(self._rotateAngelRad))+self._imgWidth*np.abs(np.cos(self._rotateAngelRad)))
            #计算大小记得绝对值

            #在仿射矩阵中把平移分量移动高度差//2这么多，以把旋转后的图片放置在新计算的大小图片中央
            M[0,2]+=(self._imgRotatedWidth-self._imgWidth)//2
            M[1,2]+=(self._imgRotatedHeight-self._imgHeight)//2
            self._M = M
        else:
            print("无原始图片")
    #旋转
    def rotate(self, angel):
        self.setRotateAngel(angel)
        self.calculateRotationData()
        print('rotation angel: ', self._rotateAngel)
        print('rotation center: ', self._rotationCenter)
        print('M: ', self._M)
        self._imgRotated = cv2.warpAffine(self._img, self._M, (self._imgRotatedWidth, self._imgRotatedHeight))


    #创建窗口
    def createWindow(self, windowName):
        cv2.namedWindow(windowName)
        self._windowName = windowName
        self._windowExisted = True
    #在窗口展示图片
    def showImgRotation(self, delayTime = 0):
        if self._windowExisted:
            cv2.imshow(self._windowName, self._imgRotated)
            cv2.waitKey(delayTime)

    #运行
    def run(self):
        #设置旋转参数
        self.setRotationCenter((self._imgWidth//2, self._imgHeight//2))
        self.createWindow("Rotation")
        for i in range(0,360,1):
            self.rotate(i)
            self.showImgRotation(10)

if __name__ == '__main__':
    test = Rotate360(os.path.join(r'C:\Users\LFK\OneDrive\Python_project\learning\pics', 'view.jpg'), True, 0.5)
    test.run()




