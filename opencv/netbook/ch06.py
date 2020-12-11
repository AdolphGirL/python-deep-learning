# -*- coding: utf-8 -*-


"""
傅立葉轉換，圖像使用二圍離散傅立葉轉換
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

file_img = os.curdir + os.sep + 'data' + os.sep + 'org.png'
img = cv2.imread(file_img, cv2.IMREAD_GRAYSCALE)
print(img.shape)

res1 = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
print(res1.shape)
dft_shift = np.fft.fftshift(res1)
print(dft_shift.shape)

res2 = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('原圖像')
plt.axis('off')
plt.subplot(122)
plt.imshow(res2, 'gray')
plt.title('轉換')
plt.axis('off')
plt.show()