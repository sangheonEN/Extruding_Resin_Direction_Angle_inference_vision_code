#%%
"""
1. gray scale image에서 내가 원하는 threshold 값 이상이면 0 미만이면 255로 되도록 저장하고 출력
단, 속도 빠르게
"""

import cv2
import numpy as np
#
# img_path = './his_equalization/case1.png'
#
# img = cv2.imread(img_path, flags=0)
#
# # ret, gray = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
# cv2.imshow('zzz', img)
#
# img[img > 130] = 255
# img[img <= 130] = 0
#
# cv2.imshow('zz', img)


def onChange(pos):
    pass

src = cv2.imread("./his_equalization/case1.png", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Trackbar Windows")

cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)

cv2.setTrackbarPos("threshold", "Trackbar Windows", 127)
cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)

while cv2.waitKey(1) != ord('q'):

    thresh = cv2.getTrackbarPos("threshold", "Trackbar Windows")
    maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")

    _, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)

    cv2.imshow("Trackbar Windows", binary)


cv2.waitKey(0)
cv2.destroyAllWindows()