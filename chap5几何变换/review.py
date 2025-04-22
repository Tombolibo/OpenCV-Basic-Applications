import cv2
import numpy as np

img = cv2.imread(r'./pics/view.jpg')
img = cv2.resize(img, None, None, 0.5, 0.5)
cols = img.shape[1]
rows = img.shape[0]

p1 = np.float32([[0,0],
    [cols-1,0],
    [0,rows-1],
    [cols-1,rows-1]]

)

alpha = 0.1

p2 = np.float32([
    [int(cols*alpha), int(rows*alpha)],
    [int(cols*(1-alpha)), int(rows*alpha)],
    [0,rows-1],
    [cols-1,rows-1]]

)

M = cv2.getPerspectiveTransform(p1, p2)
print("M:\n",M)
imgPerspective = cv2.warpPerspective(img, M, (cols, rows))

cv2.imshow('test', imgPerspective)
cv2.waitKey(0)