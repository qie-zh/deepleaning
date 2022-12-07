import cv2 as cv
import matplotlib.pyplot as plt

# 1.模版匹配，先导入两张灰度图
img = cv.imread('Photos/hasimoto.jpeg',0)
template = cv.imread('Photos/face.jpeg',0)

# TM_SQDIFF:平方差的和，越小越相关
# TM_CCORR:乘积和，越大越相关
# TM_CCOEFF:相关系数，越大越相关
# TM_SQDIFF_NORMED:归一化，越0越相关
# TM_CCORR_NORMED:越1
# TM_CCOEFF_NORMED:越1
# 尽可能使用归一化
# 输出一个与模版差异程度的矩阵。If image is W×H and templ is w×h , then result is (W−w+1)×(H−h+1) .
res = cv.matchTemplate(img,templ=template,method=cv.TM_CCOEFF_NORMED)
# 获取输出结果的最大值最小值坐标
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
# print(img.shape)
# print(template.shape)
# print(res.shape)
methods=['cv.TM_CCOEFF','cv.TM_CCOEFF_NORMED','cv.TM_SQDIFF','cv.TM_SQDIFF_NORMED','cv.TM_CCORR','cv.TM_CCORR_NORMED']

# 2.输出效果
h,w = template.shape[:2]
for meth in methods:
    img_copy = img.copy()
    method = eval(meth)

    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # TM_SQDIFF方法要取最小值，越小越相关
    if method in [cv.TM_SQDIFF,cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0]+w,top_left[1]+h)

    cv.rectangle(img_copy,top_left,bottom_right,(0,0,255),2)

    plt.subplot(121),plt.imshow(res,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(img_copy,cmap='gray')
    plt.xticks([]),plt.yticks([])
    plt.suptitle(meth)
    plt.show()
    

# cv.imshow('image',img)





# cv.waitKey(0)
# cv.destroyAllWindows()