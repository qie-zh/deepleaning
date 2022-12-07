import cv2 as cv
import numpy as np

img = cv.imread('Photos/park.jpg')

cv.imshow('park',img)

# 平移 x:右 -x：左 y：下 -y：上
def translate(img,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# translated = translate(img,-100,100)
# cv.imshow('translate',translated)

# 旋转
def rotate(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)
    
    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)#旋转矩阵，不改变大小
    dimensions = (width,height)

    return cv.warpAffine(img,rotMat,dimensions)

# rotated = rotate(img,-30)
# cv.imshow('rotate',rotated)

# 调整大小


# 翻转
# 0 左右翻转（按y轴），1上下翻转（x轴），-1上下左右翻转
flip = cv.flip(img,1)
cv.imshow('flip',flip)





cv.waitKey(0)

cv.destroyAllWindows()