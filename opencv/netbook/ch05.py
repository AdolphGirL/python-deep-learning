# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']

# 高斯濾波，img、(size,size)高斯核，sigmaX，x軸上的標準差，sigmaY，y軸上的標準差
# result = cv2.GaussianBlur(img, (5, 5), 1, 1)