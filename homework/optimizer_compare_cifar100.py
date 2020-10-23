# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np


def change_one_hot_label(translate_x):
    """
    將label轉換為on hot
    :param translate_x:
    :return:
    """
    t = np.zeros((translate_x.shape[0], 100))
    # print(t.shape)
    for idx, row in enumerate(t):
        row[translate_x[idx]] = 1

    return t


def gen_ci_far100_data(normalize=True, one_hot_label=False):
    """
    產製cifar100資料
    :param normalize: 是否正規化，將圖片資料轉換為0-1之間
    :param one_hot_label: 是否one hot encoding
    :return:
    """
    file_dir = os.pardir + os.sep + 'data' + os.sep + 'cifar-100-python'

    train_file_path = os.path.join(file_dir, 'train')
    test_file_path = os.path.join(file_dir, 'test')

    ci_far_train_dict = None
    with open(train_file_path, 'rb') as fo:
        ci_far_train_dict = pickle.load(fo, encoding='bytes')

    # print('[optimizer_compare_cifar100.py]-[gen_ci_far100_train_test_data] train_data_dict keys: {}'
    #           .format(ci_far_train_dict.keys()))

    ci_far_100_train_data = ci_far_train_dict[b'data']
    if normalize:
            ci_far_100_train_data = ci_far_100_train_data / 255.0
    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] ci_far_100_train_data shape: {}'
          .format(ci_far_100_train_data.shape))

    # fine_labels，回傳為list，封裝為ndarray
    ci_far_100_train_fine_label = np.array(ci_far_train_dict[b'fine_labels'])
    if one_hot_label:
        ci_far_100_train_fine_label = change_one_hot_label(ci_far_100_train_fine_label)
    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] ci_far_100_train_fine_label shape: {}'
          .format(ci_far_100_train_fine_label.shape))

    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] end of train data load ... ')

    ci_far_test_dict = None
    with open(test_file_path, 'rb') as fo:
        ci_far_test_dict = pickle.load(fo, encoding='bytes')

    # print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] ci_far_test_dict keys: {}'
    #       .format(ci_far_test_dict.keys()))

    ci_far_100_test_data = ci_far_test_dict[b'data']
    if normalize:
            ci_far_100_test_data = ci_far_100_test_data / 255.0
    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] ci_far_100_test_data shape: {}'
          .format(ci_far_100_test_data.shape))

    ci_far_100_test_fine_label = np.array(ci_far_test_dict[b'fine_labels'])
    if one_hot_label:
        ci_far_100_test_fine_label = change_one_hot_label(ci_far_100_test_fine_label)
    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] ci_far_100_test_fine_label shape: {}'
          .format(ci_far_100_test_fine_label.shape))

    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] end of train test load ... ')

    return (ci_far_100_train_data, ci_far_100_train_fine_label), (ci_far_100_test_data, ci_far_100_test_fine_label)


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = gen_ci_far100_data(normalize=True)

    print('[optimizer_compare_cifar100.py]-[main] train_x test: {}，train_x shape: {}'
          .format(type(train_x), train_x.shape))
    print('[optimizer_compare_cifar100.py]-[main] train_y test: {}，train_y shape: {}'
          .format(type(train_y), train_y.shape))
    print('[optimizer_compare_cifar100.py]-[main] test_x test: {}，test_x shape: {}'
          .format(type(test_x), test_x.shape))
    print('[optimizer_compare_cifar100.py]-[main] test_y test: {}，test_y shape: {}'
          .format(type(test_y), test_y.shape))
