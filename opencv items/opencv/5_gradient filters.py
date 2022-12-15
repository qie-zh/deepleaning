import cv2 as cv

img = cv.imread('Photos/1.jpg')

cv.imshow('normal',img)

# sobel算子
# gx = [[-1,0,1],
#       [-2,0,2],
#       [-1,0,1]]
#ddepth -1:和原图像深度相同,无法表示负数,cv_64f:可以保留负数。
#dx dy:保留水平或者垂直方向,分别求两个维度的值然后再求和,不建议直接计算（效果不好）
dst_x = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
dst_y = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
# cv.imshow('sobel',dst_y)

# 取绝对值，显示出负数的部分
sobelx = cv.convertScaleAbs(dst_x)
sobely = cv.convertScaleAbs(dst_y)
# cv.imshow('sobelx',sobelx)
sobelxy = cv.addWeighted(sobelx,0.5,sobely,0.5,0)
cv.imshow('sobelxy',sobelxy)

#scharr算子,比sobel算子的差值更大，更丰富的梯度信息
# gx =[[-3,0,3],
#      [-10,0,10],
#      [-3,0,3]]
scharrx = cv.Scharr(img,cv.CV_64F,1,0)
scharry = cv.Scharr(img,cv.CV_64F,0,1)
scharrx = cv.convertScaleAbs(scharrx)
scharry = cv.convertScaleAbs(scharry)

scharrxy = cv.addWeighted(scharrx,0.5,scharry,0.5,0)
cv.imshow('scharr',scharrxy)

# laplacian算子，对一些变化更敏感，但是对一些噪音也更敏感
# g = [[0,1,0],
#      [1,-4,1],
#      [0,1,0]]
laplacian = cv.Laplacian(img,cv.CV_64F)
laplacian = cv.convertScaleAbs(laplacian)
cv.imshow('laplacian',laplacian)



cv.waitKey(0)
cv.destroyAllWindows()