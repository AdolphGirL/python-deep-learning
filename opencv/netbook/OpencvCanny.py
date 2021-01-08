# -*- coding: utf-8 -*-


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']

"""
Canny = cv2.Canny(img, threshold1, threshold2, apertureSize)
"""

if __name__ == '__main__':
    file_img = os.curdir + os.sep + 'data' + os.sep + 'Lenna.jpg'
    img = cv2.imread(file_img, cv2.IMREAD_COLOR)

    # Canny
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    guess_blur = cv2.GaussianBlur(gray_img, (7, 7), 1, 1)
    Canny = cv2.Canny(guess_blur, 50, 150)

    # 顯示圖片
    Canny = cv2.cvtColor(Canny, cv2.COLOR_BGR2RGB)
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    titles = ['原圖', 'Canny-OpenCV']
    images = [img_show, Canny]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.show()