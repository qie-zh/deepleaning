import cv2 as cv

#可用于图像、视频、直播视频
def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
    #INTER_AREA 表示图像放大缩小时采样的方法，可以得到无摩尔纹的结果

#只用于调用摄像头，不可以用于已存在的视频文件
def changeRes(width,height):
    capture.set(3,width)
    capture.set(4,height)


# img = cv.imread('Photos/cat.jpg')
# cv.imshow('cat',img)

# img_resize = rescaleFrame(img)
# cv.imshow('cat1',img_resize)

# capture = cv.VideoCapture('Videos/dog.mp4')
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()

    # frame_resize = rescaleFrame(frame)
    # cv.imshow('dog',frame_resize)
    # cv.imshow('dog1',frame)

    changeRes(200,200)
    cv.imshow('live video',frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
