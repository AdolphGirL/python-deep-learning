# -*- coding: utf-8 -*-


import numpy as np


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    :param col:
    :param input_shape:
    :param filter_h:
    :param filter_w:
    :param stride:
    :param pad:
    :return:
    """
    n, c, h, w = input_shape
    out_h = (h + 2 * pad - filter_h) // stride + 1
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # col最後的輸出原為
    # col.transpose(n, out_h, out_w, c, filter_h, filter_w).reshape(n * out_h * out_w, -1)
    # 轉換為col = np.zeros((n, c, filter_h, filter_w, out_h, out_w))
    col = col.reshape(n, out_h, out_w, c, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((n, c, h + 2 * pad + stride - 1, w + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:h + pad, pad:w + pad]
