import cv2 as cv
import numpy as np

# 轮廓排序
def sort_contours(cnts):
    reverse = False
    i=0
    boundingboxes = [cv.boundingRect(c) for c in cnts]
    (cnts,boundingboxes) = zip(*sorted(zip(cnts,boundingboxes),key = lambda b: b[1][i],reverse=reverse))

    return cnts,boundingboxes

# 1.转化成灰度图
car = cv.imread('/Users/zhangheng/Desktop/deeplearning/opencv/items/carnumber/photos/car.png')

number = cv.imread('/Users/zhangheng/Desktop/deeplearning/opencv/items/carnumber/photos/number.jpg')
number_gray = cv.cvtColor(number,cv.COLOR_BGR2GRAY)
# 2.二值转化
ret,number_gray = cv.threshold(number_gray,100,255,cv.THRESH_BINARY)

# 3.轮廓检测,只检测外轮廓保留坐标点
contours, hierarchy = cv.findContours(number_gray.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

res = cv.drawContours(number,contours,-1,(0,0,255),2)
contours = sort_contours(contours)[0]
digits = {}

for (i,c) in enumerate(contours,1):
    (x,y,w,h) = cv.boundingRect(c)
    roi = number_gray[y:y+h,x:x+w]
    roi = cv.resize(roi,(57,88))
    
    digits[i] = roi


cv.imshow('number',number)
cv.imshow('car1',car)

cv.waitKey(0)
cv.destroyAllWindows()