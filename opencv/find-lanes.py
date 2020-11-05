# -*- coding: utf-8 -*-


import numpy as np
import cv2


# 比較一下兩種方式
# img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
# print(img.shape)

img_color = cv2.imread('test_image.jpg', cv2.IMREAD_COLOR)
print(img_color.shape)

# 比較一下兩種方式讀取轉換的灰階程度，一樣
img_copy = np.copy(img_color)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)

# (5, 5)的捲積核(高斯核，需為奇數)，來計算圖像高斯分布標準差
# blur = cv2.GaussianBlur(img, (5, 5), 0)，透過高斯分布
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

img_canny = cv2.Canny(img_blur, 50, 150)

cv2.imshow('org', img_gray)
# cv2.imshow('gauss', img_blur)
cv2.imshow('canny', img_canny)

cv2.waitKey(0)
cv2.destroyAllWindows()




