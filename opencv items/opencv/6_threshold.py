import cv2 as cv
import matplotlib.pyplot as plt

# 阈值处理只能使用灰度图，1通道
img = cv.imread('Photos/1.jpg',cv.IMREAD_GRAYSCALE)

cv.imshow('image',img)
# 越黑的地方数值越低，越亮的地方数值越大
# cv.THRESH_BINARY 超过阈值部分取最大值否则取0
# cv.THRESH_TRUNC 超过阈值的部分取阈值，其他不变
# cv.THRESH_TOZERO 大于阈值的部分不变，否则为0
ret, dst1 = cv.threshold(img,100,255,cv.THRESH_BINARY)
ret, dst2 = cv.threshold(img,100,255,cv.THRESH_BINARY_INV)
ret, dst3 = cv.threshold(img,100,255,cv.THRESH_TRUNC)
ret, dst4 = cv.threshold(img,100,255,cv.THRESH_TOZERO)
ret, dst5 = cv.threshold(img,100,255,cv.THRESH_TOZERO_INV)

title = ['original','binary','binary_inv','trunc','tozero','tozero_inv']
images = [img,dst1,dst2,dst3,dst4,dst5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])

plt.show()


cv.waitKey(0)
cv.destroyAllWindows()