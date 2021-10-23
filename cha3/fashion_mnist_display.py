# -*- coding: utf-8 -*-


import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import matplotlib.pyplot as plt


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img)
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def main():
    # 加载数据
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print(x_train[0])

    # 数据规模与样例
    # print(len(x_train), '\t', len(y_train))
    # print(len(x_test), '\t', len(y_test))
    # feature, label = x_train[0], y_train[0]
    # print(feature.shape, feature.dtype)
    # print(label, type(label), label.dtype)

    # # 部分数据可视化
    # X, y = [], []
    # for i in range(10):
    #     X.append(x_train[i])
    #     y.append(y_train[i])
    # show_fashion_mnist(X, get_fashion_mnist_labels(y))

    # 数据读取
    batch_size = 256
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

    start_time = time.time()
    for X, y in train_iter:
        continue
    print("%.2f sec" % (time.time() - start_time))



if __name__ == "__main__":
    main()
