# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']


file_img = os.curdir + os.sep + 'data' + os.sep + 'org.png'
img = cv2.imread(file_img)
print(img.shape)
img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_img.shape)
gray_img_show = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
print(gray_img_show.shape)

plt.subplot(121)
plt.imshow(img_show)
plt.subplot(122)
plt.imshow(gray_img_show)

plt.axis('off')
plt.show()

# 灰度處理，YUV與RGB關係，Y亮度，Y=0.3R+0.59G+0.11B
h, w = img.shape[0:2]
self_gray = np.zeros((h, w), dtype=img.dtype)
for i in range(h):
    for j in range(w):
        self_gray[i, j] = 0.3 * img[i, j, 2] + 0.59 * img[i, j, 1] + 0.11 * img[i, j, 0]

self_gray_show = cv2.cvtColor(self_gray, cv2.COLOR_GRAY2RGB)
plt.imshow(self_gray_show)
plt.axis('off')
plt.title('Y-灰階處理')
plt.show()