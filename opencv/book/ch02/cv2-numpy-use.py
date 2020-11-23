# -*- coding: utf-8 -*-


import numpy as np
import cv2
import os

img_path = os.pardir + os.sep + 'data' + os.sep + 'Lenna.jpg'
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_colo = cv2.imread(img_path)

print('img_gray shape: {}'.format(img_gray.shape))
print('img_colo shape: {}'.format(img_colo.shape))

# 透過item查詢值 item(x, y, 位置的數組索引)
print('img_colo: item 150, 120，B通道的值: {}'.format(img_colo.item(150, 120, 0)))
# 透過itemset設定值img_colo.itemset((150, 120, 0), 255)
