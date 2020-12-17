# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

"""
二值化處理，使用opencv情況

ret, threshold1 = cv2.threshold(gray, threshold, MaxP, type)

threshold1：阈值化处理的结果图像
gray：需要阈值化处理的灰度图像
threshold：需要设定的阈值
MaxP：像素最大值(默认填写255)
type：参数类型，通过输入不同的参数确定不同的阈值化处理方法(掌握该函数使用的重点)

cv2.THRESH_BINARY -------------对应二进制阈值化 (大於閥值255、小於閥值為0)
cv2.THRESH_BINARY_INV ------对应反二进制阈值化   (大於閥值0、小於閥值為255)
cv2.THRESH_TRUNC -------------对应截断阈值化   (大於閥值為閥值、小於閥值為0)
cv2.THRESH_TOZERO -----------对应阈值化为0    (大於閥值不變、小於閥值為0)
cv2.THRESH_TOZERO_INV ----对应反阈值化0       (大於閥值為0、小於閥值不變)
"""

file_img = os.curdir + os.sep + 'data' + os.sep + 'org.png'
img = cv2.imread(file_img, cv2.IMREAD_COLOR)

# 灰度處理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['灰度图像', '二进制阈值化', '反二进制阈值化', '截止阈值化', '阈值化为0', '反阈值化为0']
images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(len(titles)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
