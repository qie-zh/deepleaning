#视频处理

import cv2 as cv
import numpy as np



capture = cv.VideoCapture('3.mp4')

while True:
    isTrue , frame = capture.read()
    if not isTrue:
        print('the path is wrong')
        break

    cv.imshow('video',frame)
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()