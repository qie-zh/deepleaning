import cv2 as cv

# 滤波的作用是平滑图像，去除噪音
img = cv.imread('Photos/4.jpeg')
cv.imshow('image',img)

# 均值滤波,核中的每个值相加取平均
blur = cv.blur(img,(3,3))
# cv.imshow('blur',blur)

# 高斯滤波
# 满足高斯分布,离中心点越近权重越大
gaussian = cv.GaussianBlur(img,(5,5),1)
# cv.imshow("gaussian",gaussian)

# 中值滤波
# 卷积核中所有数的中值
median = cv.medianBlur(img,5)
cv.imshow('median',median)

cv.waitKey(0)
cv.destroyAllWindows()