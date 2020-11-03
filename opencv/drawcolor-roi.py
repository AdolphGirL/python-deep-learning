# -*- coding: utf-8 -*-


import numpy as np
import cv2


# 測試照片
img = cv2.imread("color-pick.png")

# 取得的遮罩
# img_mask = None


def pick_color(event, x, y, flags, param):
    # global img_mask
    # print('[pick_color] flags: {}'.format(flags))
    # print('[pick_color] param: {}'.format(param))

    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = img[x, y]

        # you might want to adjust the ranges(+-10, etc):
        upper = np.array([pixel[0] + 20, pixel[1] + 50, pixel[2] + 50])
        lower = np.array([pixel[0] - 20, pixel[1] - 50, pixel[2] - 50])
        print('[pick_color] pixel: {}，lower: {}，upper: {}'.format(pixel, lower, upper))

        img_mask = cv2.inRange(img, lower, upper)
        cv2.imshow("mask", img_mask)
        res = cv2.bitwise_and(img, img, mask=img_mask)
        cv2.imshow("res", res)
        # cv2.destroyAllWindows()


cv2.namedWindow('Image')
cv2.setMouseCallback('Image', pick_color)
cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
#
# print(img_mask)
#
# blurred = cv2.GaussianBlur(img_mask, (15, 15), 0)
#
# # find contours in the image
# (_, cnts, _) = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print('cnts: {}'.format(cnts))
#
# if len(cnts) > 0:
#     cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
#
#     # compute the (rotated) bounding box around then
#     # contour and then draw it
#     rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
#     cv2.drawContours(img, [rect], -1, (0, 255, 0), 2)
#
# cv2.imshow("Tracking", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
