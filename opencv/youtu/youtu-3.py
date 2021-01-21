# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def cv_show(name, img_mat):
    cv2.imshow(winname=name, mat=img_mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


file_img1 = os.curdir + os.sep + 'data' + os.sep + 'Lenna.jpg'
file_img2 = os.curdir + os.sep + 'data' + os.sep + 'ch03-02.png'

file_img1_gray = cv2.imread(file_img1, cv2.IMREAD_GRAYSCALE)

# 二值化，大於127為255，小於127為0
rect, th1 = cv2.threshold(file_img1_gray, 127, 255, type=cv2.THRESH_BINARY)
print(rect)

# 反轉上面二值化
rect, th2 = cv2.threshold(file_img1_gray, 127, 255, type=cv2.THRESH_BINARY_INV)

# 截斷，大於127，設為127(將小於閾值的灰度值設為0，大於閾值的值保持不變)
rect, th3 = cv2.threshold(file_img1_gray, 127, 255, type=cv2.THRESH_TRUNC)

# 大於127不改變，小於則為0
rect, th4 = cv2.threshold(file_img1_gray, 127, 255, type=cv2.THRESH_TOZERO)

# 反轉上述
rect, th5 = cv2.threshold(file_img1_gray, 127, 255, type=cv2.THRESH_TOZERO_INV)

titles = ['ORG', 'THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
images = [file_img1_gray, th1, th2, th3, th4, th5]
for idx in range(len(images)):
    plt.subplot(2, 3, idx + 1)
    plt.imshow(images[idx], cmap='gray')
    plt.title(titles[idx])
    plt.xticks([])
    plt.yticks([])

plt.show()
