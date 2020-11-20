# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

"""
基本IO練習
"""
img = np.zeros((3, 3), dtype=np.uint8)
print(img, img.shape)

# 利用cvtColor轉換為BGR，自動生成3個通道
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(img, img.shape)

# 默認情況下，imread會傳回BGR三個通道的資料，即使是灰階照片也會返回3個通道

# 隨機字節轉換
randomByteArray = bytearray(os.urandom(120000))
# print(randomByteArray)
flatNumpyArray = np.array(randomByteArray)
# print(flatNumpyArray)

gray_img = flatNumpyArray.reshape(300, 400)
cv2.imwrite('RandomGray.png', gray_img)

color_img = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', gray_img)
