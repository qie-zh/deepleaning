#图像处理

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('2.jpeg',1)
if img is None:
    print('the file path is wrong')

'''    
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyWindow('image')
'''
plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.xticks([]),plt.yticks([])
plt.show()

