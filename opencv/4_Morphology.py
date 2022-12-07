import cv2 as cv

img = cv.imread('Photos/4.jpeg')
cv.imshow('normal',img)

# 膨胀
dilated = cv.dilate(img,(3,3),iterations=3)
# cv.imshow('dilate',dilated)

# 侵蚀,让信息变少,interations表示重复执行的次数
eroded = cv.erode(img,(3,3),iterations=3)
# cv.imshow('eroded',eroded)

# 开运算：先腐蚀再膨胀，用于去除微小噪点
opening = cv.morphologyEx(img,cv.MORPH_OPEN,(5,5))
# cv.imshow('opening',opening)

# 闭运算
closing = cv.morphologyEx (img,cv.MORPH_CLOSE,(5,5))
# cv.imshow('closing',closing)

# 梯度运算,可以得到边界信息
gradient = cv.morphologyEx(img,cv.MORPH_GRADIENT,(3,3))
# cv.imshow('gradient',gradient)

# 礼帽,原图-开运算结果，保留噪点信息
tophat = cv.morphologyEx(img,cv.MORPH_TOPHAT,(3,3))
cv.imshow('tophat',tophat)

# 黑帽,闭运算-原图,保留轮廓 
blackhat = cv.morphologyEx(img,cv.MORPH_BLACKHAT,(3,3))
cv.imshow('blackhat',blackhat)


cv.waitKey(0)
cv.destroyAllWindows()