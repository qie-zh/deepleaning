#视频处理

import cv2 as cv
import numpy as np



def changeRes(width, heighth):
    #用于实时视频，比如 开启摄像头时
    cap.set(3,width)
    cap.set(4,heighth)

cap = cv.VideoCapture(0)
#changeRes(600,500)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True。//第一个返回值表示是否成功获取，第二个参数是获取到的一帧图像
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 我们在框架上的操作到这里
    gray = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
    # 显示结果帧e
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
# 完成所有操作后，释放捕获器
cap.release()
cv.destroyAllWindows()


'''
def rescaleframe(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    heighth = int(frame.shape[0] * scale)
    
    dimesions = (width, heighth)

    return cv.resize(frame, dimesions, interpolation=cv.INTER_AREA)

def changeRes(width, heighth):
    capture.set(3,width)
    capture.set(4,heighth)


capture = cv.VideoCapture('3.mp4')

while True:
    isTrue , frame = capture.read()
    if not isTrue:
        print('the path is wrong')
        break

    frame_resize = rescaleframe(frame)
    cv.imshow('video',frame_resize)
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
'''