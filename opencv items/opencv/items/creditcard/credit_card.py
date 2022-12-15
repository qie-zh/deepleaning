import cv2 as cv
import numpy as np

# 轮廓排序
def sort_contours(cnts):
    reverse = False
    i=0
    # 计算外接矩形，并且记录左上角的坐标
    boundingboxes = [cv.boundingRect(c) for c in cnts]
    (cnts,boundingboxes) = zip(*sorted(zip(cnts,boundingboxes),key = lambda b: b[1][i],reverse=reverse))

    return cnts,boundingboxes

# 1.转化成灰度图
number = cv.imread('/Users/zhangheng/Desktop/deeplearning/opencv/items/creditcard/photos/ocr_a_reference.png')
number_gray = cv.cvtColor(number,cv.COLOR_BGR2GRAY)

# 2.二值转化
# 不用THRESH_BINARY_INV的话会导致轮廓识别时只识别到最外围的框
ret,number_gray = cv.threshold(number_gray,100,255,cv.THRESH_BINARY_INV)

# 3.轮廓检测,只检测外轮廓保留坐标点
contours, hierarchy = cv.findContours(number_gray,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

res = cv.drawContours(number,contours,-1,(0,0,255),2)
# 4.给模版排序标号
cnts = sort_contours(contours)[0]
digits = {}

for (i,c) in enumerate(cnts):
    (x,y,w,h) = cv.boundingRect(c)
    roi = number_gray[y:y+h,x:x+w]
    roi = cv.resize(roi,(57,88))
    
    digits[i] = roi

# 5.对输入图像做处理,调整大小,灰度图
img = cv.imread('/Users/zhangheng/Desktop/deeplearning/opencv/items/creditcard/photos/credit_card_03.png')
img = cv.resize(img,(500,300))
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 进行一次高斯滤波
img_gray = cv.GaussianBlur(img_gray,(3,3),1)
cv.imshow('image_normal',img_gray)
# 礼帽,字节创建的核需要和自己想保留的信息相关
recKernel = cv.getStructuringElement(cv.MORPH_RECT,(9,3))
tophat = cv.morphologyEx(img_gray,cv.MORPH_TOPHAT,recKernel)

# sobel算子,突出信息
gradX = cv.Sobel(tophat,ddepth=cv.CV_64F,dx=1,dy=0,ksize=-1)
gradY = cv.Sobel(tophat,cv.CV_64F,0,1,-1)
gradXY = cv.addWeighted(gradX,0.5,gradY,0.5,0)

# 取绝对值
gradXY = np.absolute(gradXY)

# 归一化
(minVal,maxVal) = (np.min(gradXY),np.max(gradXY))
gradXY = (255*((gradXY - minVal)/(maxVal - minVal))) 
gradXY = gradXY.astype('uint8')

# 闭操作，先膨胀再腐蚀，让数字连在一起
gradXY = cv.morphologyEx(gradXY,cv.MORPH_CLOSE,recKernel)

# cv.imshow('sobel',gradXY)
# 阈值处理，用otsu或者triang的方法时可以自动选择合适的阈值，并不是拿0作为基准
thresh = cv.threshold(gradXY,0,255,cv.THRESH_BINARY | cv.THRESH_TRIANGLE)[1]
cv.imshow('contours',thresh)
# 再来一次闭操作，减少数字间的空隙,效果不明显迭代两次
thresh = cv.morphologyEx(thresh,cv.MORPH_CLOSE,recKernel)
# 计算轮廓
threshcnts, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
img_copy = img.copy()
cv.drawContours(img_copy,threshcnts,-1,(0,0,255,3),2)
# cv.imshow('contours',thresh)
# 筛选轮廓
locs = []

for (i,c) in enumerate(threshcnts):
    (x,y,w,h) = cv.boundingRect(c)
    ar = w / float(h)

    if ar > 2.5 and ar < 4.0:
         if (w>70 and w < 90) and (h > 20 and h <35) and y>150:
            locs.append((x,y,w,h))
# (57, 152, 76, 22) (157, 152, 77, 22) (357, 151, 77, 23) (257, 151, 77, 23) 图片3的区域
# (362, 156, 80, 24) (259, 155, 79, 25) (156, 155, 79, 25) (54, 155, 78, 25) 图片4
# (49, 166, 84, 29) (361, 165, 87, 32) (257, 165, 84, 31)(154, 165, 89, 33) 图片5

# i = 5
# (x1,y1,w1,h1) = locs[i]
# cv.rectangle(img,(x1,y1),(x1+w1,y1+h1),(0,0,255),2)
# cv.imshow('rectangle',img)
# print(locs[i])

locs = sorted(locs,key=lambda x: x[0])
output = []

for (i,(gx,gy,gw,gh)) in enumerate(locs):
    groupOutput = []
    # 摘出感兴趣区域
    group = img_gray[gy-5:gy+gh+5,gx-5:gx+gw+5]
    
    # 二值化
    group_thr = cv.threshold(group,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    # 进行一次滤波轮廓识别的效果更好
    # group_thr = cv.blur(group_thr,(3,3))
    # cv.imshow('image1',group_thr)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 检测数字框中每个数字的轮廓
    digitcnts, hierarchy = cv.findContours(group_thr,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # 给轮廓排序
    digitcnts = sort_contours(digitcnts)[0]

    # cv.drawContours(group,digitcnts,-1,255,2)
    # cv.imshow('image',group)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


    # 单独处理框中的每个数字
    for c in digitcnts:
        # 给数字外接矩阵
        (x,y,w,h) = cv.boundingRect(c)
        roi = group_thr[y:y+h,x:x+w]
        roi = cv.resize(roi,(57,88))

        # 框出每个数字
        # cv.rectangle(group,(x,y),(x+w,y+h),(0,0,255),2)
        # cv.imshow('group',group)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        scores = []
        # 模版匹配
        for (digit,digitROI) in digits.items():
            result = cv.matchTemplate(roi,digitROI,cv.TM_CCOEFF)

            score = cv.minMaxLoc(result)[1]
            scores.append(score)
        
        groupOutput.append(str(np.argmax(scores)))

        cv.rectangle(img,(gx-5,gy-5),(gx+gw+5,gy+gh+5),(0,0,255),2)

        cv.putText(img,''.join(groupOutput),(gx,gy-15),cv.FONT_HERSHEY_SIMPLEX,0.85,(0,0,255),2)

        output.extend(groupOutput)


cv.imshow('result',img)
cv.waitKey(0)
cv.destroyAllWindows()