import cv2
import numpy as np

import chineseImgReader

img = chineseImgReader.imgRead(r'./pics/cell.jpg')

g0 = img.copy()
g1 = cv2.pyrDown(g0)
g2 = cv2.pyrDown(g1)
g3 = cv2.pyrDown(g2)

l1 = cv2.subtract(g0,cv2.pyrUp(g1))
l2 = cv2.subtract(g1, cv2.pyrUp(g2))
l3 = cv2.subtract(g2,cv2.pyrUp(g3))


cv2.imshow('g0',g0)
cv2.imshow('g1',g1)
cv2.imshow('g2',g2)
cv2.imshow('g3',g3)
cv2.imshow('l1',l1)
cv2.imshow('l2',l2)
cv2.imshow('l3',l3)
cv2.waitKey()