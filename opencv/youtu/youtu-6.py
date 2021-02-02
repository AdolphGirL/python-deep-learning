# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np


def cv_show(name, img_mat):
    cv2.imshow(winname=name, mat=img_mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


file_img = os.curdir + os.sep + 'data' + os.sep + 'Lenna.jpg'
img1 = cv2.imread(file_img, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(file_img)

img_org = np.hstack((img1, img2[:, :, 0]))
cv_show('before', img_org)

img3 = cv2.Canny(img1, 50, 100)
img4 = cv2.Canny(img1, 80, 250)
img5 = np.hstack((img3, img4))
cv_show('after', img5)

img3 = cv2.Canny(img2, 50, 100)
img4 = cv2.Canny(img2, 80, 250)
img5 = np.hstack((img3, img4))
cv_show('after', img5)
