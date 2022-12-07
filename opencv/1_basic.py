import cv2 as cv
import matplotlib.pyplot as plt

#导入三通道颜色的图片（bgr）
img = cv.imread('Photos/1.jpg')
# cv.imshow('image_normal',img)

# 颜色转换，灰色
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

# 模糊,元组的值越大模糊效果越明显
# blur = cv.GaussianBlur(img,(5,5),cv.BORDER_DEFAULT)
# cv.imshow('blur',blur)

# 边缘检测,可以通过更改阈值来改变边缘数量，也可以导入模糊后的图像
# canny = cv.Canny(img,100,175)
# cv.imshow('canny',canny)



# 改变大小
# 关于插值的方法 INTRER_AREA用于缩小，CUBIC用于缩放效果最好但是慢，LINEAR用于缩放
# resized = cv.resize(img,(500,500),interpolation=cv.INTER_AREA)
# 按比例放大图像
resized = cv.resize(img,(0,0),fx=2,fy=4)
cv.imshow('resized',resized)

# 裁剪
# 利用数组切片
# cropped = img[:,::2]
# cv.imshow('crop',cropped)
# print('%d %d'%(img.shape[0],img.shape[1]))


# 边界填充
'''
cat_img = cv.imread('Photos/cats.jpg')

top_size,bottom_size,left_size,right_size=(50,50,50,50)

#复制最边缘像素
replicate = cv.copyMakeBorder(cat_img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_REPLICATE)
#镜像复制
reflect = cv.copyMakeBorder(cat_img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_REFLECT)
#镜像复制，但不复制边缘像素
reflect01 = cv.copyMakeBorder(cat_img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_REFLECT_101)
#拼接图片
wrap = cv.copyMakeBorder(cat_img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_WRAP)
#填充常量
constant = cv.copyMakeBorder(cat_img,top_size,bottom_size,left_size,right_size,borderType=cv.BORDER_CONSTANT,value=0)

plt.subplot(231),plt.imshow(cv.cvtColor(cat_img,cv.COLOR_BGR2RGB)),plt.title('original')
plt.subplot(232),plt.imshow(cv.cvtColor(replicate,cv.COLOR_BGR2RGB)),plt.title('replicate')
plt.subplot(233),plt.imshow(cv.cvtColor(reflect,cv.COLOR_BGR2RGB)),plt.title('reflect')
plt.subplot(234),plt.imshow(cv.cvtColor(reflect01,cv.COLOR_BGR2RGB)),plt.title('reflect01')
plt.subplot(235),plt.imshow(cv.cvtColor(wrap,cv.COLOR_BGR2RGB)),plt.title('wrap')
plt.subplot(236),plt.imshow(cv.cvtColor(constant,cv.COLOR_BGR2RGB)),plt.title('constant')

plt.show()
'''

#图像融合,图像大小需要一样
# cat1_img = cv.imread('Photos/cat.jpg')
# cat2_img = cv.imread('Photos/cats 2.jpg')

# # print(cat1_img.shape)
# # print(cat2_img.shape)
# gamma:偏移量,加权之后的图像上再加一个常数,不要太大
# res = cv.addWeighted(cat1_img,0.4,cat2_img,0.6,0)
# cv.imshow('add image',res)

cv.waitKey(0)
cv.destroyAllWindows()