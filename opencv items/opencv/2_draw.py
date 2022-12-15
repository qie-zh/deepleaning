import cv2 as cv
import numpy as np

blank = np.zeros((300,300,3),dtype='uint8')#类型不能乱写

# 1.给一个区域画颜色

# blank[0:100,0:100] = 0,255,0

# 2.画一个矩形的框

#颜色通道为蓝绿红,thickness表示线的宽度，负数表示填满区域
# cv.rectangle(blank,(0,0),(100,100),(0,0,255),thickness=-1)

#shape[0]表示多少行，shape[1]表示多少列
# cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0),thickness=-1)

# 3.画一个圆
# 指定圆心和半径
# cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,0,255),thickness=-1)

# 4.画一条线
# 直线宽度不能为-1
# cv.line(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(255,0,255),thickness=3)

# 5.写文本
cv.putText(blank,'hello',(0,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,255),thickness=2)

cv.imshow('Blank',blank)

cv.waitKey(0)