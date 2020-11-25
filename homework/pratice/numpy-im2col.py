# -*- coding: utf-8 -*-


import numpy as np


# pad函式測試
# A = np.arange(1, 5).reshape(2, 2)
# print(A)
#
# B = np.pad(A, ((1, 1), (2, 2)), 'constant')
# print(B)


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    :param input_data: 由(資料量, 通道, 高, 長)的4維陣列構成的輸入資料
    :param filter_h: 卷積核的高
    :param filter_w: 卷積核的長
    :param stride: 步幅
    :param pad: 填充
    :return: 2維陣列
    """
    n, c, h, w = input_data.shape

    # 輸出資料的高
    out_h = (h + 2 * pad - filter_h) // stride + 1

    # 輸出資料的長
    out_w = (w + 2 * pad - filter_w) // stride + 1

    # 填充 H, W，pad第二個參數為每個軸要填充的情況，設定為(前，後)
    # input_data有四個軸，前面兩個不需要填充
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)],
                 'constant', constant_values=(0, 0))
    # 等同
    # img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # (N, C, filter_h, filter_w, out_h, out_w)的0矩陣，後續再轉置
    col = np.zeros((n, c, filter_h, filter_w, out_h, out_w))

    # 取出輸入數據上，根據卷積核大小，將三個通道依次展開為一維數組
    # 然後再連接為一個長的一維數組，再根據步輻，
    # 將輸入數據中每個應用卷積核的地方都會生成一個一維數組，共N個(批次的數量)
    # 所以基本輸出的第一軸會為N*out_h*out_w，第二軸會為每行的元素數，C*filter_h*filter_w
    for y in range(filter_h):
        # 取得行區間
        y_max = y + stride * out_h
        for x in range(filter_w):
            # 取得列區間
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 按(0, 4, 5, 1, 2, 3)順序，交換col的列，然後改變形狀
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, -1)
    return col


if __name__ == '__main__':
    t = np.random.rand(3, 3, 4, 4)
    t = im2col(t, 2, 2, 2, 0)
    print(t)
    print()
    print(t.shape)



