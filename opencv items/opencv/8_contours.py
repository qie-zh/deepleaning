import cv2 as cv
import numpy as np

# 1.找到轮廓并输出轮廓
# 导入灰度图
img = cv.imread('Photos/shapes.jpeg')
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 进行二值处理
ret, threshold = cv.threshold(img_gray,100,255,cv.THRESH_OTSU)

# 执行轮廓检测
contours, hierarchy = cv.findContours(threshold,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

# 绘制轮廓
# 创建纯黑背景，大小同原图，如果线条有颜色不可以为灰度图
blank = np.zeros(img.shape,dtype='uint8')

# image: 目标图像(以该图片为背景) contours:获取的轮廓 
# contoursidx:指定的轮廓（负数表示全部）color:线条颜色 tickness:线条粗细
res = cv.drawContours(blank,contours,-1,(0,0,255),2)
# 执行完会更改目标图像,如果要保留原图像可以使用.copy()
# print(cv.contourArea(contours[0]))# 打印该轮廓的面积
# print(cv.arcLength(contours[0],True))# 打印周长
# cv.imshow('image',res)


# 2.轮廓近似
# 用直线替代曲线或者接近
shapes = cv.imread('Photos/strange shapes.jpeg')
shapes_gray = cv.cvtColor(shapes,cv.COLOR_BGR2GRAY)
ret, threshold1 = cv.threshold(shapes_gray,100,255,cv.THRESH_OTSU)
contours1, hierarchy1 = cv.findContours(threshold1,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
cnt = contours1[1]
blank1 = np.zeros(shapes.shape,dtype='uint8')
# res1 = cv.drawContours(blank1,cnt,-1,(0,0,255),2)

# 近似精度,0.1倍的周长来做阈值
espilon = 0.1* cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,espilon,True)
res1 = cv.drawContours(shapes,[approx],-1,(0,0,255),2)
# cv.imshow('shapes',res1)

# 3.外接图形
x,y,w,h = cv.boundingRect(cnt)
rectangle = cv.rectangle(res1,(x,y),(x+w,y+h),(0,255,0),2)

# cv.imshow('image',rectangle)



cv.waitKey(0)
cv.destroyAllWindows()