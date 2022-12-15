import cv2 as cv
import pytesseract

img = cv.imread('/Users/zhangheng/Desktop/deeplearning/opencv/Photos/screen shot.png')

text = pytesseract.image_to_string(img)
print(text)