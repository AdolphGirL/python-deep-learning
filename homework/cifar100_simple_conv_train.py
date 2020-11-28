# -*- coding: utf-8 -*-
import numpy as np
from collections import OrderedDict
from common.layers import *
import os
import pickle
import matplotlib.pyplot as plt
from common.trainer import Trainer


class CifarSimpleConvNet:
    """
    for CIFAR100，兩個卷積層，一個全連階層，一個輸出層
    conv - relu - pool - conv - relu - pool - affine - relu - affine - softmax

    feature map仿LeNet-5
                conv1                       pool                        conv
    N*3*32*32 -> 6*3*5*5 -> 6*3*28*28 -> 3*2*2,stride=2 -> 6*3*14*14 -> 16*3*3*3 -> 16*3*12*12

        pool                          flatten
    -> 3*2*2,stride=2 -> 16*3*6*6 -> 16*6*6 -> fc input = 576 -> 256 -> 100(ouput)
    """
    def __init__(self, input_dim=(3, 32, 32),
                 conv_param={'filter_num_1': 6, 'filter_size_1': 5, 'pad_1': 0, 'stride_1': 1,
                             'filter_num_2': 16, 'filter_size_2': 3, 'pad_2': 0, 'stride_2': 1},
                 hidden_size=256, output_size=100, weight_init_std=0.01):
        """
        :param input_dim:
        :param conv_param:
        :param hidden_size_1:
        :param hidden_size_2:
        :param output_size:
        :param weight_init_std:
        """
        filter_num_1 = conv_param['filter_num_1']
        filter_size_1 = conv_param['filter_size_1']
        filter_pad_1 = conv_param['pad_1']
        filter_stride_1 = conv_param['stride_1']

        filter_num_2 = conv_param['filter_num_2']
        filter_size_2 = conv_param['filter_size_2']
        filter_pad_2 = conv_param['pad_2']
        filter_stride_2 = conv_param['stride_2']

        input_size = input_dim[1]

        # 此處假定輸入的長寬都一樣，且假定pooling層2*2，stride=2
        # 卷積、池化(池化也可以padding)計算方式相同，N = (W − F + 2P ) / S + 1
        conv_output_size_1 = (input_size - filter_size_1 + 2 * filter_pad_1) / filter_stride_1 + 1
        pool_output_size_1 = (conv_output_size_1 - 2 + 2 * 0) / 2 + 1
        conv_output_size_2 = (pool_output_size_1 - filter_size_2 + 2 * filter_pad_2) / filter_stride_2 + 1
        pool_output_size_2 = (conv_output_size_2 - 2 + 2 * 0) / 2 + 1

        # 第一個全連階層的輸入
        fc_input_size = int(filter_num_2 * pool_output_size_2 * pool_output_size_2)

        # 參數建立為後續演算需要，conv參數建立為(FN, C, H, W)
        self.params = {
            'W1': weight_init_std * np.random.randn(filter_num_1, input_dim[0], filter_size_1, filter_size_1),
            'b1': np.zeros(filter_num_1),
            'W2': weight_init_std * np.random.randn(filter_num_2, filter_num_1, filter_size_2, filter_size_2),
            'b2': np.zeros(filter_num_2),
            'W3': weight_init_std * np.random.randn(fc_input_size, hidden_size),
            'b3': np.zeros(hidden_size),
            'W4': weight_init_std * np.random.randn(hidden_size, output_size),
            'b4': np.zeros(output_size)
        }

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride_1'], conv_param['pad_1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride_2'], conv_param['pad_2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()

        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """
        prediction function
        :param x:
        :return:
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        loss function
        :param x: input parameter data
        :param t: teacher label
        :return:
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """
        precision arithmetic
        :param x:
        :param t:
        :param batch_size:
        :return:
        """
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        return acc / x.shape[0]

    def gradient(self, x, t):
        """
        solving gradient reverse the spread
        :param x:
        :param t:
        :return:
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # set up
        grads = {
            'W1': self.layers['Conv1'].dW, 'b1': self.layers['Conv1'].db,
            'W2': self.layers['Conv2'].dW, 'b2': self.layers['Conv2'].db,
            'W3': self.layers['Affine1'].dW,
            'b3': self.layers['Affine1'].db,
            'W4': self.layers['Affine2'].dW,
            'b4': self.layers['Affine2'].db
        }

        return grads


def gen_cifar_data(if_logger: bool = False):
    """
    讀取cifar100資料
    :return:
    """
    file_dir = os.pardir + os.sep + 'data' + os.sep + 'cifar-100-python'

    train_file_path = os.path.join(file_dir, 'train')
    test_file_path = os.path.join(file_dir, 'test')

    ci_far_train_dict = None
    with open(train_file_path, 'rb') as fo:
        ci_far_train_dict = pickle.load(fo, encoding='bytes')

    # print('[cifar100_simple_conv_train.py]-[gen_cifar_data] train_data_dict keys: {}'
    #           .format(ci_far_train_dict.keys()))

    ci_far_100_train_data = ci_far_train_dict[b'data']
    if if_logger:
        print('[cifar100_simple_conv_train.py]-[gen_cifar_data] ci_far_100_train_data shape: {}'
              .format(ci_far_100_train_data.shape))

    ci_far_100_train_fine_label = np.array(ci_far_train_dict[b'fine_labels'])
    if if_logger:
        print('[cifar100_simple_conv_train.py]-[gen_cifar_data] ci_far_100_train_fine_label shape: {}'
              .format(ci_far_100_train_fine_label.shape))

    print('[cifar100_simple_conv_train.py]-[gen_cifar_data] end of train data load ... ')

    ci_far_test_dict = None
    with open(test_file_path, 'rb') as fo:
        ci_far_test_dict = pickle.load(fo, encoding='bytes')

    # print('[cifar100_simple_conv_train.py]-[gen_cifar_data] ci_far_test_dict keys: {}'
    #       .format(ci_far_test_dict.keys()))

    ci_far_100_test_data = ci_far_test_dict[b'data']
    if if_logger:
        print('[cifar100_simple_conv_train.py]-[gen_cifar_data] ci_far_100_test_data shape: {}'
              .format(ci_far_100_test_data.shape))

    ci_far_100_test_fine_label = np.array(ci_far_test_dict[b'fine_labels'])
    if if_logger:
        print('[cifar100_simple_conv_train.py]-[gen_cifar_data] ci_far_100_test_fine_label shape: {}'
              .format(ci_far_100_test_fine_label.shape))

    print('[cifar100_simple_conv_train.py]-[gen_cifar_data] end of train test load ... ')

    return (ci_far_100_train_data, ci_far_100_train_fine_label), (ci_far_100_test_data, ci_far_100_test_fine_label)


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = gen_cifar_data(False)

    # 轉換train_x shape還原為未壓平狀況，且dataset資料格式為32*32*3
    train_x = train_x.reshape(-1, 32, 32, 3)
    test_x = test_x.reshape(-1, 32, 32, 3)

    # 在轉換為layer定義的格式
    train_x = train_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)

    print('[cifar100_simple_conv_train.py]-[main] train_x test: {}，train_x shape: {}'
          .format(type(train_x), train_x.shape))
    print('[cifar100_simple_conv_train.py]-[main] train_y test: {}，train_y shape: {}'
          .format(type(train_y), train_y.shape))
    print('[cifar100_simple_conv_train.py]-[main] test_x test: {}，test_x shape: {}'
          .format(type(test_x), test_x.shape))
    print('[cifar100_simple_conv_train.py]-[main] test_y test: {}，test_y shape: {}'
          .format(type(test_y), test_y.shape))

    # 減少一下樣本數
    x_train, t_train = train_x[:5000], train_y[:5000]
    x_test, t_test = test_x[:1000], test_y[:1000]

    max_epochs = 20

    # 先採用預設情況
    network = CifarSimpleConvNet()

    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()