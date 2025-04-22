import cv2
import numpy as np

img = cv2.imread(r'./pics/view.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros(imgGray.shape, dtype = np.uint8)
mask[100:200,100:200] = 255
img = cv2.bitwise_or(img, img, mask=mask)

cv2.imshow('test', img)
cv2.waitKey(0)