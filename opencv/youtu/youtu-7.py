# -*- coding: utf-8 -*-

# ref: https://www.mdeditor.tw/pl/gkhv/zh-tw
# ref: https://github.com/Y132om/Credit_card_identification


# opencv的簡化版
from imutils import contours
import numpy as np
import cv2
import os


tem_path = os.curdir + os.sep + 'data' + os.sep + 'tem.png'
che_path = os.curdir + os.sep + 'data' + os.sep + 'check.png'


def cv_show(name, show_img):
    cv2.imshow(name, show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 重置大小，用于比较模板和图像中的数字是否一致
# 插值方法如下：
# INTER_NEAREST:最邻近插值
# INTER_LINEAR:双线性插值,默认情况下使用该方式进行插值.
# INTER_AREA:基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法,它可以产生更少的波纹,
# 但是当图像放大时,它的效果与INTER_NEAREST效果相似.
# INTER_CUBIC:4×4邻域双3次插值
# INTER_LANCZOS4:8×8邻域兰索斯插值
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]  # (200,300,3)
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 排序模板
def sort_contours(cnt_list, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    '''
    cv2.boundingRect(c) 
    返回四个值，分别是x，y，w，h；
    x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    '''
    bounding_boxes = [cv2.boundingRect(c) for c in cnt_list]
    # cnts的第一個對應到bounding_boxes的第一個
    # 将对象中对应的元素打包成一个个元组，排序看使用x軸或者y軸的值
    (cnt_list, bounding_boxes) = zip(*sorted(zip(cnt_list, bounding_boxes), key=lambda b: b[1][i], reverse=reverse))
    return cnt_list, bounding_boxes


# 模板處理
img = cv2.imread(tem_path)
cv_show('template', img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('template_gray', img_gray)

# 二值化，cv2.THRESH_BINARY_INV黑白反轉(原圖黑色為字體，反轉後黑會變白，才有數值)
ref = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('template_bi', ref)

# 計算輪廓
# cv2.findContours()函式接受的引數為二值圖，
# 即黑白的（不是灰度圖）,cv2.RETR_EXTERNAL只檢測外輪廓，
# cv2.CHAIN_APPROX_SIMPLE只保留終點座標(只保留各个轮廓的部分顶点或者转折点，足够用来描绘出轮廓)
# 返回的ref_cnts，list中每個元素都是影象中的一個輪廓(存放著每一個輪廓的點位(顶点或者转折点))
ref_cnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 看一下排序前
# for (i, c) in enumerate(ref_cnts):
#     (x, y, w, h) = cv2.boundingRect(c)
#     # 取出圖
#     roi = ref[y:y + h, x:x + w]
#     # roi = cv2.resize(roi, (57, 88))
#     cv_show(str(i), roi)


# 查看點位內容
# boundingRect(x，y，w，h；x，y是矩阵左上点的坐标，w，h是矩阵的宽和高)
# bounding_boxes = [cv2.boundingRect(cnt) for cnt in ref_cnts]
# for bbox in bounding_boxes:
#     print(bbox)

# 在原圖畫一下，看看有沒有差很多
# 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。后面的参数很简单
# 剩餘的引數是顏色(0 B, 0 G, 255 R)、厚度
cv2.drawContours(img, ref_cnts, -1, (0, 0, 255), 3)
cv_show('template_Contours', img)
print('[youtu-7] 模板輪廓的數目，ref_cnts shape: {}'.format(np.array(ref_cnts, dtype=object).shape))
#
# 排序，從左到右，從上到下
ref_cnts = sort_contours(ref_cnts, method="left-to-right")[0]

# 看一下排序後
# for (i, c) in enumerate(ref_cnts):
#     (x, y, w, h) = cv2.boundingRect(c)
#     # 取出圖
#     roi = ref[y:y + h, x:x + w]
#     # roi = cv2.resize(roi, (57, 88))
#     cv_show(str(i), roi)

# 紀錄排序後模板的數字
digits = {}
for (i, c) in enumerate(ref_cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    # 放大一下
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi
print('[youtu-7] 排序後模板的數字，digits: {}'.format(digits))

# 輸入影像
img_input = cv2.imread(che_path)
cv_show('Input_img', img_input)
img_input = resize(img_input, width=300)
gray_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
cv_show('Input_gray', gray_input)

# 初始化卷積核,根據實際任務指定大小,不一定非要3x3
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 禮帽操作，突出更明亮的區域
# 形態學操作,禮帽+閉操作可以突出明亮區域,但並不是非得禮帽+閉操作
tophat = cv2.morphologyEx(gray_input, cv2.MORPH_TOPHAT, rectKernel)
cv_show('Input_tophat', tophat)

# ksize=-1相當於用3*3的
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
