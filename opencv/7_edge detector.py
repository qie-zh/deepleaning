import cv2 as cv
import numpy as np

# 边缘检测(需要灰度图)流程：
# 1.高斯滤波
# 2.计算像素点梯度和方向
# 3.非极大值抑制
# 4.双阈值
# 5.抑制孤立的弱边缘

img = cv.imread('Photos/2.jpeg',cv.IMREAD_GRAYSCALE)

img1 = cv.Canny(img,80,150)
img2 = cv.Canny(img,50,100)

res = np.hstack((img1,img2))

cv.imshow('image',res)
cv.waitKey(0)
cv.destroyAllWindows()