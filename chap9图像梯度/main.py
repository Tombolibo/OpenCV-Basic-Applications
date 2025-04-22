import cv2
import numpy as np

img = cv2.imread(r'./pics/way.jpg')
img = cv2.resize(img, None, None, 0.7, 0.7)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgSobelX_16S = cv2.Sobel(img, cv2.CV_16S, 1,0)
imgSobelY_16S = cv2.Sobel(img, cv2.CV_16S, 0,1)
print(imgSobelX_16S.min(), imgSobelX_16S.max())

# #区间映射方式
# imgSobelX_16S_8U = (imgSobelX_16S-np.iinfo(np.int16).min)/(np.iinfo(np.int16).max-np.iinfo(np.int16).min)*255
# imgSobelX_16S_8U = imgSobelX_16S_8U.astype(np.uint8)
#最大值最小值映射方式
# imgSobelX_16S_8U = (imgSobelX_16S-imgSobelX_16S.min())/(imgSobelX_16S.max()-imgSobelX_16S.min())*255
# imgSobelX_16S_8U = imgSobelX_16S_8U.astype(np.uint8)
#截断方式
imgSobelX_16S_8U = cv2.convertScaleAbs(imgSobelX_16S,None, 1,0)
imgSobelY_16S_8U = cv2.convertScaleAbs(imgSobelY_16S, None, 1,0)
# imgSobel_8U = cv2.add(imgSobelY_16S_8U, imgSobelX_16S_8U)
imgSobel_8U = cv2.addWeighted(imgSobelX_16S_8U, 0.5, imgSobelY_16S_8U,0.5,0)

cv2.imshow('scaleAbs X', imgSobelX_16S_8U)
cv2.imshow('scaleAbs Y', imgSobelY_16S_8U)
cv2.imshow('scaleAbs', imgSobel_8U)
cv2.waitKey(0)



imgScharrX_16S = cv2.Scharr(img, cv2.CV_16S, 1,0)
imgScharrY_16S = cv2.Scharr(img, cv2.CV_16S, 0,1)
print(imgScharrX_16S.min(), imgScharrX_16S.max())

# #区间映射方式
# imgSobelX_16S_8U = (imgSobelX_16S-np.iinfo(np.int16).min)/(np.iinfo(np.int16).max-np.iinfo(np.int16).min)*255
# imgSobelX_16S_8U = imgSobelX_16S_8U.astype(np.uint8)
#最大值最小值映射方式
# imgSobelX_16S_8U = (imgSobelX_16S-imgSobelX_16S.min())/(imgSobelX_16S.max()-imgSobelX_16S.min())*255
# imgSobelX_16S_8U = imgSobelX_16S_8U.astype(np.uint8)
#截断方式
imgScharrX_16S_8U = cv2.convertScaleAbs(imgScharrX_16S,None, 1,0)
imgScharrY_16S_8U = cv2.convertScaleAbs(imgScharrY_16S, None, 1,0)
# imgScharr_8U = cv2.add(imgScharrY_16S_8U, imgScharrX_16S_8U)
imgScharr_8U = cv2.addWeighted(imgScharrX_16S_8U, 0.5, imgScharrY_16S_8U, 0.5, 0)

cv2.imshow('Scharr X', imgScharrX_16S_8U)
cv2.imshow('Scharr Y', imgScharrY_16S_8U)
cv2.imshow('ScharrABS', imgScharr_8U)
cv2.waitKey(0)


#拉普拉斯
imgLaplacian = cv2.Laplacian(img, cv2.CV_16S, None, 3)
imgLaplacian = cv2.convertScaleAbs(imgLaplacian)
cv2.imshow('Laplacian', imgLaplacian)
cv2.waitKey(0)