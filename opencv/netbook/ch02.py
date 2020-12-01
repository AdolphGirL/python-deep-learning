# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']


file_img = os.curdir + os.sep + 'data' + os.sep + 'org.png'
img = cv2.imread(file_img)
# print(img.shape)
img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray_img.shape)
gray_img_show = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
# print(gray_img_show.shape)

# plt.subplot(121)
# plt.imshow(img_show)
# plt.subplot(122)
# plt.imshow(gray_img_show)
#
# plt.axis('off')
# plt.show()

# 灰度處理，YUV與RGB關係，Y亮度，Y=0.3R+0.59G+0.11B
h, w = img.shape[0:2]
self_gray = np.zeros((h, w), dtype=img.dtype)
for i in range(h):
    for j in range(w):
        self_gray[i, j] = 0.3 * img[i, j, 2] + 0.59 * img[i, j, 1] + 0.11 * img[i, j, 0]

self_gray_show = cv2.cvtColor(self_gray, cv2.COLOR_GRAY2RGB)
# plt.imshow(self_gray_show)
# plt.axis('off')
# plt.title('Y-灰階處理')
# plt.show()

# 最大值灰度(即3個通道的最大值為像素值)
max_gray = np.zeros((h, w), dtype=img.dtype)
for i in range(h):
    for j in range(w):
        max_gray[i, j] = max(img[i, j, 0], img[i, j, 1], img[i, j, 2])

max_gray_show = cv2.cvtColor(max_gray, cv2.COLOR_GRAY2RGB)
# plt.imshow(max_gray_show)
# plt.axis('off')
# plt.title('最大值灰階處理')
# plt.show()

# 平均值灰度，方式雷同最大值灰度

# Gamma灰度，參閱https://blog.csdn.net/qq_42451251/article/details/107783243
gamma_gray = np.zeros((h, w), dtype=img.dtype)
for i in range(h):
    for j in range(w):
        # 分子
        a = img[i, j, 2] ** 2.2 + 1.5*img[i, j, 1] ** 2.2 + 0.6 * img[i, j, 0] ** 2.2
        # 分母
        b = 1 + 1.5 ** 2.2 + 0.6 ** 2.2
        # 開2.2次方根
        gamma_gray[i, j] = pow(a/b, 1.0/2.2)

gamma_gray_show = cv2.cvtColor(gamma_gray, cv2.COLOR_GRAY2RGB)

# 展示所有灰階圖
plt.subplot(151)
plt.imshow(img_show)
plt.axis('off')
plt.title('原圖')

plt.subplot(152)
plt.imshow(gray_img_show)
plt.axis('off')
plt.title('opencv-灰階')

plt.subplot(153)
plt.imshow(self_gray_show)
plt.axis('off')
plt.title('Y-灰階')

plt.subplot(154)
plt.imshow(max_gray_show)
plt.axis('off')
plt.title('最大值灰階')

plt.subplot(155)
plt.imshow(gamma_gray_show)
plt.axis('off')
plt.title('gamma')

plt.show()
