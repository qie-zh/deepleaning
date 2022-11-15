import sys

import cv2 as cv
import matplotlib
import numpy as np
'''
print(sys.version)
print(cv.__version__)
print(np.__version__)
print(matplotlib.__version__)
'''

img = cv.imread('1.jpg',0)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()