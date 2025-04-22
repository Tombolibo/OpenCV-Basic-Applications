import cv2
import numpy as np


class MyBilateral():
    def __init__(self):
        self.imgSrc = None
        self.imgShow = None
        self.windowName = None
        self.sigmaSpace = 0
        self.sigmaColor = 0
    def readImg(self, fileName):
        if fileName is not None:
            self.imgSrc = cv2.imread(fileName)
    def copyImg(self):
        if self.imgSrc is not None:
            self.imgShow = self.imgSrc.copy()
    def createWindow(self, windowName):
        if windowName is not None:
            self.windowName = windowName
            cv2.namedWindow(self.windowName)
    def createTrack(self):
        if self.windowName is not None:
            cv2.createTrackbar('sigmaColor', self.windowName, 0,255,self.setSigmaColor)
            cv2.createTrackbar('sigmaSpace', self.windowName, 0,100,self.setSigmaSpace)
    def setSigmaSpace(self, v):
        self.sigmaSpace = v
        self.bilateralImg()
    def setSigmaColor(self, v):
        self.sigmaColor = v
        self.bilateralImg()
    def bilateralImg(self):
        if self.imgSrc is not None and self.sigmaColor>0 and self.sigmaSpace>0:
            print(self.imgSrc.shape, self.sigmaSpace, self.sigmaColor)
            self.imgShow = cv2.bilateralFilter(self.imgSrc,-1,self.sigmaColor, self.sigmaSpace)
            print('Done')

    def run(self):
        self.readImg(r'./pics/qb.png')
        self.createWindow('my bilateral')
        self.copyImg()
        self.createTrack()
        keycode = -1
        while(keycode != 27):
            if keycode == 32:
                self.copyImg()
            else:
                cv2.imshow(self.windowName, self.imgShow)
            keycode = cv2.waitKey(1)

if __name__ == '__main__':
    a = MyBilateral()
    a.run()
