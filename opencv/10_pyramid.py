import cv2 as cv

img = cv.imread('Photos/group 1.jpg')
cv.imshow('image',img)
# 1.高斯金字塔: 先用高斯滤波，然后再采样
# 向上采样(扩充图像使图像变大)
up = cv.pyrUp(img)
# cv.imshow('up',up)

# 向下采样(缩小图像)
down = cv.pyrDown(img)
cv.imshow('down',down)

# 2.拉普拉斯金字塔


cv.waitKey(0)
cv.destroyAllWindows()