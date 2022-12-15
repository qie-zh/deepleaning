#读取图片和视频

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
# img = cv.imread('Photos/2.jpeg',1)
# if img is None:
#     print('the file path is wrong')

# cv.imshow('image',img)
# cv.waitKey(0)
# cv.destroyWindow('image')

# 读取视频
capture = cv.VideoCapture('Videos/dog.mp4')
while True:
    isTrue ,frame = capture.read()
    cv.imshow('video',frame)
    #运行完成后会报错，因为每20帧的播放最后会找不到图像
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
