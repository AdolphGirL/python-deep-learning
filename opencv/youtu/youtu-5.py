# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


"""
cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
第一個引數是需要處理的影象；
第二個引數是影象的深度，-1表示採用的是與原影象相同的深度。目標影象的深度必須大於等於原影象的深度；
dx和dy表示的是求導的階數，0表示這個方向上沒有求導，一般為0、1、2。
其後是可選的引數：

dst不用解釋了；
ksize是Sobel運算元的大小，必須為1、3、5、7。
scale是縮放導數的比例常數，預設情況下沒有伸縮係數；
delta是一個可選的增量，將會加到最終的dst中，同樣，預設情況下沒有額外的值加到dst中；
borderType是判斷影象邊界的模式。這個引數預設值為cv2.BORDER_DEFAULT。

在Sobel函式的第二個引數這裡使用了cv2.CV_16S。
因為OpenCV文件中對Sobel運算元的介紹中有這麼一句：
“in the case of 8-bit input images it will result in truncated derivatives”。
即Sobel函式求完導數後會有負值，還有會大於255的值。
而原影象是uint8，即8位無符號數，所以Sobel建立的影象位數不夠，會有截斷。
因此要使用16位有符號的資料型別，即cv2.CV_16S。

在經過處理後，別忘了用convertScaleAbs()函式將其轉回原來的uint8形式。
否則將無法顯示影象，而只是一副灰色的視窗。convertScaleAbs()的原型為：
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])

其中可選引數alpha是伸縮係數，beta是加到結果上的一個值。結果返回uint8型別的圖片。
由於Sobel運算元是在兩個方向計算的，最後還需要用cv2.addWeighted(...)函式將其組合起來。
其函式原型為：
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
其中alpha是第一幅圖片中元素的權重，beta是第二個的權重，gamma是加到最後結果上的一個值。
"""


def cv_show(name, img_mat):
    cv2.imshow(winname=name, mat=img_mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


file_img = os.curdir + os.sep + 'data' + os.sep + 'Lenna.jpg'
img = cv2.imread(file_img, cv2.IMREAD_GRAYSCALE)
img_x = cv2.Sobel(img, -1, 1, 0, ksize=3)
img_y = cv2.Sobel(img, -1, 0, 1, ksize=3)

cv_show('x', img_x)
cv_show('y', img_y)


img_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
# 轉回uint8，沒有負值
absX = cv2.convertScaleAbs(img_x)
img_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
absY = cv2.convertScaleAbs(img_y)

cv_show('x', absX)
cv_show('y', absY)

# dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])
# gamma -> 偏值項，通常為0即可
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
cv_show('all', dst)
