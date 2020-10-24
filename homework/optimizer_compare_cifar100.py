# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from common.optimizer import *
from common.multi_layer_net import MultiLayerNet
from common.util import smooth_curve
import matplotlib.pyplot as plt


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

    train_dict = None
    with open(train_file_path, 'rb') as fo:
        train_dict = pickle.load(fo, encoding='bytes')

    # print('[optimizer_compare_cifar100.py]-[gen_ci_far100_train_test_data] train_dict keys: {}'
    #           .format(train_dict.keys()))

    train_data = train_dict[b'data']
    if normalize:
        train_data[:] = train_data[:] / 255.0
    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] train_data shape: {}'
          .format(train_data.shape))

    # fine_labels，回傳為list，封裝為ndarray
    train_label = np.array(train_dict[b'fine_labels'])
    if one_hot_label:
        train_label = change_one_hot_label(train_label)
    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] train_label shape: {}'
          .format(train_label.shape))

    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] end of train data load ... ')

    test_dict = None
    with open(test_file_path, 'rb') as fo:
        test_dict = pickle.load(fo, encoding='bytes')

    # print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] test_dict keys: {}'
    #       .format(test_dict.keys()))

    test_data = test_dict[b'data']
    if normalize:
        test_data[:] = test_data[:] / 255.0
    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] test_data shape: {}'
          .format(test_data.shape))

    test_label = np.array(test_dict[b'fine_labels'])
    if one_hot_label:
        test_label = change_one_hot_label(test_label)
    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] test_label shape: {}'
          .format(test_label.shape))

    print('[optimizer_compare_cifar100.py]-[gen_ci_far100_data] end of train test load ... ')

    return (train_data, train_label), (test_data, test_label)


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

    """
    60000筆訓練資料，batch: 200，max_iterations: 250；一個epoch
    """
    train_size = train_x.shape[0]
    batch_size = 200
    max_iterations = 250

    # 建構優化器
    optimizers = {'SGD': SGD(),
                  'Momentum': Momentum(),
                  'AdaGrad': AdaGrad(),
                  'Adam': Adam()}

    # 建構分別的網路架構，並且記錄每個batch的損失情況
    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        # 輸入32 * 32 * 3；輸出100
        networks[key] = MultiLayerNet(
            input_size=32*32*3, hidden_size_list=[1024, 512, 256], output_size=100)
        train_loss[key] = []

    for i in range(1, max_iterations + 1):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_x[batch_mask]
        t_batch = train_y[batch_mask]

        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch, t_batch)
            train_loss[key].append(loss)

        # 每一萬筆，看一下損失情況
        if i % 50 == 0:
            print("[optimizer_compare_cifar100.py]-[main] =========== iteration: {} ===========".format(i))
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print("[optimizer_compare_cifar100.py]-[main] {}: {}".format(key, loss))
            print("[optimizer_compare_cifar100.py]-[main] =========== iteration: {} END ===========".format(i))

    markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    # plt.ylim(0, 1)
    plt.legend()
    plt.show()
