import cv2
from PIL import Image
import numpy as np


#中文路径图片读取，返回nparray，8位BGR三通道
class imgReader(object):
    #类属性

    #读取类方法
    @classmethod
    def imgRead(cls, filePath):
        if filePath is not None:
            imgSrc = Image.open(filePath)
            imgArr = np.array(imgSrc)
            if imgSrc.format == 'PNG':
                return cv2.cvtColor(imgArr, cv2.COLOR_RGBA2BGR)
            elif imgSrc.format == 'JPEG':
                return cv2.cvtColor(imgArr, cv2.COLOR_RGB2BGR)
            else:
                print('格式不支持')
                return None
        else:
            print('检查文件路径')
            return None

if __name__ == '__main__':
    img = imgReader.imgRead('./pics/book.jpg')
    print(img.shape)
    cv2.imshow('test', img)
    cv2.waitKey(0)