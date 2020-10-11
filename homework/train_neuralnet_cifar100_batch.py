# -*- coding: utf-8 -*-

from common.functions import *
from common.gradient import numerical_gradient as ng
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


class ThreeLayerNet:

    """
    FOR CIFAR100測試使用，目前先定義三層網路
    input[b, 3072] => w1[3072, 1024] => w2[1024, 256] => w3[256, 100] => output[b, 100]
    """
    def __init__(self, hidden_layer_size: tuple, weight_init_std=0.01):
        """
        初始化權重參數
        :param hidden_layer_size:   隱藏層節點與層數，ex:((3072, 1024), (1024, 256), (256, 100))
        :param weight_init_std:     權重初始化標準差
        """
        self.params = {}
        self.hidden_layer_num = len(hidden_layer_size)

        print('[train_neuralnet_cifar100_batch.py]-[ThreeLayerNet] ThreeLayerNet init weight，get init value: {}'
              .format(hidden_layer_size))
        for index, item in enumerate(hidden_layer_size):
            self.params['W' + str(index + 1)] = weight_init_std * np.random.randn(item[0], item[1])
            self.params['b' + str(index + 1)] = np.zeros(item[1])

    def predict(self, predict_run_x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(predict_run_x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        return y

    def loss(self, loss_x, t):
        y = self.predict(loss_x)
        return cross_entropy_error(y, t)

    def accuracy(self, predict_x, t):
        y = self.predict(predict_x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(predict_x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        計算梯度，目前的設計會跑很久，cifar100的input已經為3072
        common.gradient import numerical_gradient，為一個元素一個元素慢慢計算梯度
        :param x:   輸入
        :param t:   結果
        :return:
        """
        # loss_w = lambda W: self.loss(x, t)
        # grads = {}
        # for index in range(self.hidden_layer_num):
        #     grads['W' + str(index)] = ng(loss_w, self.params['W' + str(index)])
        #     grads['b' + str(index)] = ng(loss_w, self.params['b' + str(index)])
        #
        # return grads
        pass

    def gradient(self, x, t):
        """
        反向傳播的梯度下降，之後再改寫為多層架構
        :param x:   輸入
        :param t:   結果
        :return:
        """
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)

        # backward
        dy = (y - t) / batch_num

        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)

        dz2 = np.dot(dy, W3.T)
        da2 = sigmoid_grad(a2) * dz2
        grads['W2'] = np.dot(z1.T, da2)
        grads['b2'] = np.sum(da2, axis=0)

        dz1 = np.dot(da2, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


def change_one_hot_label(x):
    t = np.zeros((x.shape[0], 100))
    print(t.shape)
    for idx, row in enumerate(t):
        row[x[idx]] = 1

    return t


def gen_ci_far100_train_test_data():
    """
    未處理異常情況
    :return: ((train_x, train_y), (test_x, test_y))
    """
    file_dir = os.pardir + os.sep + 'data' + os.sep + 'cifar-100-python'

    train_file_path = os.path.join(file_dir, 'train')
    test_file_path = os.path.join(file_dir, 'test')

    ci_far_train_dict = None
    with open(train_file_path, 'rb') as fo:
        ci_far_train_dict = pickle.load(fo, encoding='bytes')

    # print('[train_neuralnet_cifar100_batch.py]-[gen_ci_far100_train_test_data] train_data_dict keys: {}'
    #           .format(ci_far_train_dict.keys()))

    ci_far_100_train_data = ci_far_train_dict[b'data']
    print('[train_neuralnet_cifar100_batch.py]-[gen_ci_far100_train_test_data] ci_far_100_train_data shape: {}'
          .format(ci_far_100_train_data.shape))

    ci_far_100_train_fine_label = np.array(ci_far_train_dict[b'fine_labels'])
    ci_far_100_train_fine_label = change_one_hot_label(ci_far_100_train_fine_label)
    print('[train_neuralnet_cifar100_batch.py]-[gen_ci_far100_train_test_data] ci_far_100_train_fine_label shape: {}'
          .format(ci_far_100_train_fine_label.shape))

    print('[train_neuralnet_cifar100_batch.py]-[gen_ci_far100_train_test_data] end of train data load ... ')

    ci_far_test_dict = None
    with open(test_file_path, 'rb') as fo:
        ci_far_test_dict = pickle.load(fo, encoding='bytes')

    # print('[train_neuralnet_cifar100_batch.py]-[gen_ci_far100_train_test_data] ci_far_test_dict keys: {}'
    #       .format(ci_far_test_dict.keys()))

    ci_far_100_test_data = ci_far_test_dict[b'data']
    print('[train_neuralnet_cifar100_batch.py]-[gen_ci_far100_train_test_data] ci_far_100_test_data shape: {}'
          .format(ci_far_100_test_data.shape))

    ci_far_100_test_fine_label = np.array(ci_far_test_dict[b'fine_labels'])
    ci_far_100_test_fine_label = change_one_hot_label(ci_far_100_test_fine_label)
    print('[train_neuralnet_cifar100_batch.py]-[gen_ci_far100_train_test_data] ci_far_100_test_fine_label shape: {}'
          .format(ci_far_100_test_fine_label.shape))

    print('[train_neuralnet_cifar100_batch.py]-[gen_ci_far100_train_test_data] end of train test load ... ')

    return (ci_far_100_train_data, ci_far_100_train_fine_label), (ci_far_100_test_data, ci_far_100_test_fine_label)


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = gen_ci_far100_train_test_data()
    print('[train_neuralnet_cifar100_batch.py]-[main] train_x test: {}，train_x shape: {}'
          .format(type(train_x), train_x.shape))
    print('[train_neuralnet_cifar100_batch.py]-[main] train_y test: {}，train_y shape: {}'
          .format(type(train_y), train_y.shape))
    print('[train_neuralnet_cifar100_batch.py]-[main] test_x test: {}，test_x shape: {}'
          .format(type(test_x), test_x.shape))
    print('[train_neuralnet_cifar100_batch.py]-[main] test_y test: {}，test_y shape: {}'
          .format(type(test_y), test_y.shape))

    print('[train_neuralnet_cifar100_batch.py]-[main] start run process ... ')

    # input[b, 3072] => w1[3072, 1024] => w2[1024, 256] => w3[256, 100] => output[b, 100]
    network = ThreeLayerNet(((3072, 1024), (1024, 256), (256, 100)))

    # 根據電腦自行調整
    # 500 * 100 = train_x.shape[0] = 1epoch
    iter_num = 2500

    train_size = train_x.shape[0]
    batch_size = 100
    learning_rate = 0.1

    test_acc_list = []
    train_loss_list = []
    train_acc_list = []

    # 目前程式亂數取batch資料，因此當iter_per_epoch==train_size / batch_size
    # 可以視為執行完一次epoch
    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_x[batch_mask]
        t_batch = train_y[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in network.params.keys():
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            # 根據電腦自行調整
            # 50000筆資料太大，記憶體會爆掉，改取10000筆
            batch_train_mask = np.random.choice(train_size, 10000)
            train_x_batch = train_x[batch_train_mask]
            train_y_batch = train_y[batch_train_mask]
            train_acc = network.accuracy(train_x_batch, train_y_batch)
            train_acc_list.append(train_acc)

            test_acc = network.accuracy(test_x, test_y)
            test_acc_list.append(test_acc)
            print("[train_neuralnet_cifar100_batch.py]-[main] process {} epoch "
                  "train acc: {}，test acc: {}".format(i / iter_per_epoch, train_acc, test_acc))

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()