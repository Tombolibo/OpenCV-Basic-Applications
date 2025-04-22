import cv2
import numpy as np

img = cv2.imread(r'./pics/cell.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t, img = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)

contours, heriarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))  # 所有轮廓
print(contours[0])  # 第一个轮廓
print(contours[0][0])  # 第一个轮廓的第一个点
print(contours[0][0][0])  # 第一个轮廓的第一个点的第一行
print(contours[0][0][0][0])  # 第一个轮廓的第一个点的第一行的第一个坐标（x）