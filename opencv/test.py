import cv2 as cv
import numpy as np

img = cv.imread('Photos/1.jpg')

zeros = np.zeros((200,200,3))

cv.imshow('image',zeros)
cv.waitKey(0)
cv.destroyAllWindows()