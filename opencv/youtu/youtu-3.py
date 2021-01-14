# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def cv_show(name, img_mat):
    cv2.imshow(winname=name, mat=img_mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


file_img1 = os.curdir + os.sep + 'data' + os.sep + 'Lenna.jpg'
file_img2 = os.curdir + os.sep + 'data' + os.sep + 'ch03-02.png'



