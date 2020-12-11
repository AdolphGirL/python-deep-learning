# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']


"""
無論是numpy實現，還是OpenCV實現，
得到的結果中頻率為0的部分都會在左上角，
通常要轉換到中心位置，可以通過shift變換來實現

np.fft.fft2() 可以對訊號進行快速傅立葉變換
該函式的輸出結果是一個複數陣列complex ndarray
np.fft.fftshift() 函式將中心位置轉移至中間

(越亮（灰度值越高）的位置代表該頻率的資訊振幅越大)


numpy 影像傅立葉變換主要使用的函式如下所示
------------------------------------------------------
#實現影像逆傅立葉變換，返回一個複數陣列
numpy.fft.ifft2(a, n=None, axis=-1, norm=None)

#fftshit()函式的逆函式，它將頻譜影像的中心低頻部分移動至左上角
numpy.fft.fftshift()

#將複數轉換為0至255範圍
img = numpy.abs(逆傅立葉變換結果)
------------------------------------------------------


numpy 傅立葉逆變換
------------------------------------------------------
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)
------------------------------------------------------

"""


def high_pass_filtering(img, size):
    """
    高通濾波器
    :param img: 頻域圖
    :param size: 濾波的大小
    :return:
    """
    h, w = img.shape[0: 2]
    # 中心點
    h1, w1 = int(h/2), int(w/2)
    # 中心點加減濾波器的一半，形成濾波大小，並且像素質設為0
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0
    return img


def low_pass_filtering(img, size):
    """
    低通濾波器
    產生一個和原圖一樣大小的黑色區域，將中間部分設定為1，然後和耀濾波的圖相乘(周圍圍0，中間圈塊為1，相成就保留中間區塊的值)
    :param img:
    :param size:
    :return:
    """
    h, w = img.shape[0: 2]
    h1, w1 = int(h / 2), int(w / 2)
    img_filter = np.zeros((h, w), np.uint8)
    img_filter[h1 - int(size / 2):h1 + int(size / 2), w1 - int(size / 2):w1 + int(size / 2)] = 1
    img_res = img_filter * img
    return img_res


if __name__ == '__main__':
    # 高通濾波器，讀取照片轉換為灰階
    file_img = os.curdir + os.sep + 'data' + os.sep + 'org.png'
    img1 = cv2.imread(file_img, cv2.IMREAD_GRAYSCALE)
    # print('[main] img1.shape: {}'.format(img1.shape))

    # 傅立葉轉換
    f = np.fft.fft2(img1)
    # 將左上角低頻部分移動到中間
    shift = np.fft.fftshift(f)

    # 經過高通濾波器的轉換
    img2 = high_pass_filtering(shift, 50)
    # print('[main] img2.shape: {}'.format(img2.shape))

    # fft 結果是複數，其絕對值結果是振幅
    # 再取log轉換，可以看到光譜的能量
    res = np.log(np.abs(img2))
    # res = np.abs(img2)

    # 傅立葉逆變換
    r_shift = np.fft.ifftshift(img2)
    img3 = np.fft.ifft2(r_shift)
    # fft 結果是複數，其絕對值結果是振幅，離閃點，不須再使用log
    img3 = np.abs(img3)
    # print('[main] img3.shape: {}'.format(img3.shape))

    plt.subplot(131), plt.imshow(img1, 'gray'), plt.title('原圖像')
    plt.axis('off')

    plt.subplot(132), plt.imshow(res, 'gray'), plt.title('高通滤波')
    plt.axis('off')

    plt.subplot(133), plt.imshow(img3, 'gray'), plt.title('濾波後的圖')
    plt.axis('off')

    plt.show()
