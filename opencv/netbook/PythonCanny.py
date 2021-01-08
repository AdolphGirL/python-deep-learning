# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']


def canny(gray, min_t, max_t):
    h, w = gray.shape[0:2]

    # 高斯平滑去躁
    guess_array = np.zeros((h, w), dtype=gray.dtype)
    guess_matrix = np.array([[1, 4, 7, 4, 1],
                             [4, 16, 26, 16, 4],
                             [7, 26, 41, 26, 7],
                             [4, 16, 26, 16, 4],
                             [1, 4, 7, 4, 1]])
    guess_kernel = guess_matrix * (1 / 273)
    for i in range(2, h - 2):
        for j in range(2, w - 2):
            kernel_sum = np.sum(gray[i - 2:i + 2 + 1, j - 2:j + 2 + 1] * guess_kernel)
            guess_array[i, j] = kernel_sum

    # Sobel
    sobel_array = np.zeros((h, w), dtype=gray.dtype)
    for i in range(h - 1):
        for j in range(w - 1):
            dx = (int(guess_array[i - 1, j - 1]) + 2 * int(guess_array[i - 1, j]) + int(guess_array[i - 1, j + 1])) - \
                 (int(guess_array[i + 1, j - 1]) + 2 * int(guess_array[i + 1, j]) + int(guess_array[i + 1, j + 1]))

            dy = (int(guess_array[i - 1, j + 1]) + 2 * int(guess_array[i, j + 1]) + int(guess_array[i + 1, j + 1])) - \
                 (int(guess_array[i - 1, j - 1]) + 2 * int(guess_array[i, j - 1]) + int(guess_array[i + 1, j - 1]))
            sobel_array[i, j] = np.sqrt(dx ** 2 + dy ** 2)

    # NMS
    suppression = np.zeros((h, w), dtype=gray.dtype)
    for i in range(h - 1):
        for j in range(w - 1):
            dx = (int(guess_array[i - 1, j - 1]) + 2 * int(guess_array[i - 1, j]) + int(guess_array[i - 1, j + 1])) - \
                 (int(guess_array[i + 1, j - 1]) + 2 * int(guess_array[i + 1, j]) + int(guess_array[i + 1, j + 1]))
            dy = (int(guess_array[i - 1, j + 1]) + 2 * int(guess_array[i, j + 1]) + int(guess_array[i + 1, j + 1])) - \
                 (int(guess_array[i - 1, j - 1]) + 2 * int(guess_array[i, j - 1]) + int(guess_array[i + 1, j - 1]))

            dx = np.maximum(dx, 1e-10)
            seta = np.arctan(dy / dx)

            # 尋找梯度角度
            if -0.4142 < seta < 0.4142:
                angle = 0
            elif 0.4142 < seta < 2.4142:
                angle = 45
            elif abs(seta) > 2.4142:
                angle = 90
            elif -2.4142 < seta < -0.4142:
                angle = 135

            # 根據梯度方向求得NMS
            if angle == 0:
                if max(sobel_array[i, j], sobel_array[i, j - 1], sobel_array[i, j + 1]) == sobel_array[i, j]:
                    suppression[i, j] = sobel_array[i, j]
                else:
                    suppression[i, j] = 0
            elif angle == 45:
                if max(sobel_array[i, j], sobel_array[i - 1, j + 1], sobel_array[i + 1, j - 1]) == sobel_array[i, j]:
                    suppression[i, j] = sobel_array[i, j]
                else:
                    suppression[i, j] = 0
            elif angle == 90:
                if max(sobel_array[i, j], sobel_array[i - 1, j], sobel_array[i + 1, j]) == sobel_array[i, j]:
                    suppression[i, j] = sobel_array[i, j]
                else:
                    suppression[i, j] = 0
            elif angle == 135:
                if max(sobel_array[i, j], sobel_array[i - 1, j - 1], sobel_array[i + 1, j + 1]) == sobel_array[i, j]:
                    suppression[i, j] = sobel_array[i, j]
                else:
                    suppression[i, j] = 0

    # 雙閥值
    double_threshold = np.zeros((h, w), dtype=gray.dtype)
    for i in range(h):
        for j in range(w):
            if suppression[i, j] >= max_t:
                double_threshold[i, j] = 255
            elif suppression[i, j] <= min_t:
                double_threshold[i, j] = 0
            else:
                # 介於高界線與低界線:若附近有兩點高於高界線的點，則此點也視為邊緣
                if max(suppression[i - 1, j - 1], suppression[i - 1, j], suppression[i - 1, j + 1],
                       suppression[i, j - 1], suppression[i, j + 1], suppression[i + 1, j - 1],
                       suppression[i + 1, j + 1]) >= suppression[i, j]:
                    double_threshold[i, j] = 255
                else:
                    double_threshold[i, j] = 0
    return double_threshold


if __name__ == '__main__':
    file_img = os.curdir + os.sep + 'data' + os.sep + 'Lenna.jpg'
    img = cv2.imread(file_img, cv2.IMREAD_COLOR)

    # canny演算，threshold 2:1 - 3:1之間
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Canny = canny(gray_img, 50, 150)

    # 顯示圖片
    Canny = cv2.cvtColor(Canny, cv2.COLOR_BGR2RGB)
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    titles = ['原圖', 'Canny']
    images = [img_show, Canny]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.show()
