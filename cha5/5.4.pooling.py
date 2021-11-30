# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.zeros((X.shape[0] - p_h + 1,
                  X.shape[1] - p_w + 1))
    Y = tf.Variable(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i+p_h, j: j+p_w]))
            elif mode == 'avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i+p_h, j: j+p_w]))

    return Y


def main():
    X = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), 'avg'))
    

if __name__ == "__main__":
    main()
