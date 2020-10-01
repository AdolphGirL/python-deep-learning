# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from common.functions import sigmoid, softmax

# np.random.seed(1)

# 順練權重暫時自行生成，暫時給定-1 - 1之間
# b都給zero
# input[b, 3072] => w1[3072, 2048] => w2[2048, 1024] => w3[1024, 512] => w4[512, 256] => w5[256, 100] => output[b, 100]
network = {
    'W1': 2 * np.random.random((3072, 2048)) - 1,
    'b1': np.zeros([2048]),
    'W2': 2 * np.random.random((2048, 1024)) - 1,
    'b2': np.zeros([1024]),
    'W3': 2 * np.random.random((1024, 512)) - 1,
    'b3': np.zeros([512]),
    'W4': 2 * np.random.random((512, 256)) - 1,
    'b4': np.zeros([256]),
    'W5': 2 * np.random.random((256, 100)) - 1,
    'b5': np.zeros([100])
}

print('[neuralnet_cifar100_batch.py] W1 shape: {}'.format(network['W1'].shape))
print('[neuralnet_cifar100_batch.py] b1 shape: {}'.format(network['b1'].shape))
print('[neuralnet_cifar100_batch.py] W2 shape: {}'.format(network['W2'].shape))
print('[neuralnet_cifar100_batch.py] b2 shape: {}'.format(network['b2'].shape))
print('[neuralnet_cifar100_batch.py] W3 shape: {}'.format(network['W3'].shape))
print('[neuralnet_cifar100_batch.py] b3 shape: {}'.format(network['b3'].shape))
print('[neuralnet_cifar100_batch.py] W4 shape: {}'.format(network['W4'].shape))
print('[neuralnet_cifar100_batch.py] b4 shape: {}'.format(network['b4'].shape))
print('[neuralnet_cifar100_batch.py] W5 shape: {}'.format(network['W5'].shape))
print('[neuralnet_cifar100_batch.py] b5 shape: {}'.format(network['b5'].shape))


def get_test_data(cifar_path):
    with open(cifar_path, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


def predict(net_work, x):
    w1, w2, w3, w4, w5 = net_work['W1'], net_work['W2'], net_work['W3'], net_work['W4'], net_work['W5']
    b1, b2, b3, b4, b5 = net_work['b1'], net_work['b2'], net_work['b3'], net_work['b4'], net_work['b5']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    z3 = sigmoid(a3)
    a4 = np.dot(z3, w4) + b4
    z4 = sigmoid(a4)
    a5 = np.dot(z4, w5) + b5

    # 最後一層使用softmax分類
    y = softmax(a5)
    return y


if __name__ == '__main__':
    file_dir = os.pardir + os.sep + 'data' + os.sep + 'cifar-100-python'
    test_file_path = os.path.join(file_dir, 'test')

    # CIFAR 100
    # Each image comes with a "fine" label (the class to which it belongs)
    # and a "coarse" label (the superclass to which it belongs).
    test_data_dict = get_test_data(test_file_path)
    print('[neuralnet_cifar100_batch.py] test_data_dict keys: {}'.format(test_data_dict.keys()))

    # data資料已flatten => 32 * 32 * 3，可以直接網路
    cifar_100_data = test_data_dict[b'data']
    print('[neuralnet_cifar100_batch.py] cifar_100_data shape: {}'.format(cifar_100_data.shape))

    cifar_100_fine_label = test_data_dict[b'fine_labels']
    print('[neuralnet_cifar100_batch.py] cifar_100_fine_label len: {}'.format(len(cifar_100_fine_label)))

    # 暫時不需要
    # cifar_100_coarse_label = test_data_dict[b'coarse_labels']
    # print('cifar_100_coarse_label len: {}'.format(len(cifar_100_coarse_label)))

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, cifar_100_data.shape[0], batch_size):
        x_batch = cifar_100_data[i:i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == cifar_100_fine_label[i:i + batch_size])
        # print('[neuralnet_cifar100_batch.py] batch count: {}，accuracy_cnt: {}'.format(i, accuracy_cnt))

    print('[neuralnet_cifar100_batch.py] accuracy_cnt: {}'.format(accuracy_cnt))
    print('[neuralnet_cifar100_batch.py] Accuracy: {}'.format(float(accuracy_cnt) / float(cifar_100_data.shape[0])))






