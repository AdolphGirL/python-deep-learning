# -*- coding: utf-8 -*-

"""
0: 妙蛙種子, 1:小火龍, 2:超夢, 3:皮卡丘, 4:傑尼龜
"""
import numpy as np
import matplotlib.pyplot as plt
import os


file_dir = os.pardir + os.sep + 'data' + os.sep + 'youthai-competition'
np_train_file = os.path.join(file_dir, 'pokemon_train.npy')

train = np.load(np_train_file)
train_x, train_y = train[:, 1:], train[:, 0]
print(type(train_x), train_x.shape)
print(type(train_y), train_y.shape)


# fig, axes = plt.subplots(5, 5)
# 等價
# fig = plt.figure()
# ax = plt.subplot(111)

fig = plt.figure(figsize=(16, 10))
for i in range(5):
    idx = '15' + str(i+1)
    ax = plt.subplot(int(idx))
    plt.imshow(train_x[i].reshape(128, 128, 3))
    plt.title(train_y[i])

plt.show()


