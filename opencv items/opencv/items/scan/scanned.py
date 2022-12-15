import cv2 as cv
import numpy as np
import pytesseract


# 改变图像大小
def resize(img,width=None,heigh=None):

    (h,w) = img.shape[:2]
    if width is None and heigh is None:
        return img
    if width is None:
        r = heigh / float(h)
        dim = (int(w*r),heigh)
    else:
        r = width/float(w)
        dim = (width,int(h*r))
    resized = cv.resize(img,dim,cv.INTER_AREA)
    return resized

def imshow(cha,img):
    cv.imshow(cha,img)
    cv.waitKey(0)
    cv.destroyAllWindows()


img = cv.imread('/Users/zhangheng/Desktop/deeplearning/opencv/items/scan/photos/3.jpeg')
# 记录形状变化比例
radio = img.shape[0] / 500
img_copy = img.copy()

# 改变图片大小
img = resize(img_copy,heigh=500)

# 图片预处理,灰度图 滤波 边缘检测
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img_gray = cv.GaussianBlur(img_gray,(3,3),1)
# img_thre = cv.threshold(img_gray,100,255,cv.THRESH_TOZERO_INV)[1]
# img_close = cv.morphologyEx(img_thre,cv.MORPH_GRADIENT,(3,3))
edged = cv.Canny(img_gray,75,200)

# imshow('img',img_thre)

# 轮廓检测
cnts = cv.findContours(edged,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)[0]
# edged_copy = edged.copy()
# cv.drawContours(edged_copy,cnts[:5],-1,255,2)
# imshow('contours',edged_copy)

# 给轮廓排序
cnts = sorted(cnts,key=cv.contourArea,reverse=True)[:5]

# cv.drawContours(edged_copy,cnts,-1,255,2)
# imshow('contours',edged_copy)

# 开始处理每个轮廓
for c in cnts:
    # 轮廓近似
    peri = cv.arcLength(c,True) # 周长
    approx = cv.approxPolyDP(c,peri*0.02,True)

    # 如果近似结果得到四个点
    if len(approx) == 4:
        screenCnt = approx
        break


# cv.drawContours(img,[screenCnt],-1,(0,0,255),2)
# imshow('img',img)

# 透视变换
def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")
    
    # 下面这个操作，是因为这四个点目前是乱序的，下面通过了一种巧妙的方式来找到对应位置
    # 左上和右下， 对于左上的这个点，(x,y)坐标和会最小， 对于右下这个点，(x,y)坐标和会最大，所以坐标求和，然后找最小和最大位置就是了
    # 按照顺序找到对应坐标0123分别是左上， 右上， 右下，左下
    s = pts.sum(axis=1)
    # 拿到左上和右下
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # 右上和左下， 对于右上这个点，(x,y)坐标差会最小，因为x很大，y很小， 而左下这个点， x很小，y很大，所以坐标差会很大
    # 拿到右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    # 拿到正确的左上，右上， 右下，左下四个坐标点的位置
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 计算输入的w和h值 这里就是宽度和高度，计算方式就是欧几里得距离，坐标对应位置相减平方和开根号
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 变换后对应坐标位置   
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # 计算变换矩阵  变换矩阵这里，要有原始的图像的四个点的坐标， 变换之后的四个点的对应坐标，然后求一个线性矩阵，相当于每个点通过一个线性映射
    # 到新的图片里面去。那么怎么求线性矩阵呢？  
    # 其实解线性方程组， 原始图片四个点坐标矩阵A(4, 2)， 新图片四个点坐标矩阵B(4, 2)， 在列这个维度上扩充1维1
    # A变成了(4, 3), B也是(4, 3)， 每个点相当于(x, y, 1) 
    # B = WA， 其中W是3*3的矩阵，A的每个点是3*1， B的每个点是3*1
    # W矩阵初始化[[h11, h12, h13], [h21, h22, h23], [h31, h32, 1]]  这里面8个未知数，通过上面给出的4个点
    # 所以这里A， B四个点的坐标都扩充了1列，已知A,B四个点的坐标，这里去求参数，解8个线性方程组得到W，就是cv2.getPerspectiveTransform干的事情
    # 这个文章说的不错：https://blog.csdn.net/overflow_1/article/details/80330835
    W = cv.getPerspectiveTransform(rect, dst)
    # 有了透视矩阵W, 对于原始图片中的每个坐标， 都扩充1列，然后与W乘， 就得到了在变换之后图片的坐标点(x, y, z), 然后把第三列给去掉(x/z, y/z)就是最终的坐标
    warped = cv.warpPerspective(image, W, (maxWidth, maxHeight))

    # 返回变换后结果
    return warped

# 获得仿射变换后的图像
warped = four_point_transform(img_copy,screenCnt.reshape(4,2) * radio)
# 处理变换后的图像,二值化
warped = cv.cvtColor(warped,cv.COLOR_BGRA2GRAY)
result = cv.threshold(warped,0,255,cv.THRESH_OTSU | cv.THRESH_BINARY)[1]

# imshow('warped',result)

rows, cols = result.shape[:2]
center = (cols/2, rows/2)  # 以图像中心为旋转中心
angle = 90                 # 顺时针旋转90°
scale = 1                  # 等比例旋转，即旋转后尺度不变    
 
# M = cv.getRotationMatrix2D(center, angle, scale)
# rotated_img = cv.warpAffine(result, M, (cols, rows))

rotated_img = result
resized_result = resize(rotated_img,heigh=600)

text = pytesseract.image_to_string(rotated_img,lang='jpn+eng')
print(text)

imshow('result',resized_result)