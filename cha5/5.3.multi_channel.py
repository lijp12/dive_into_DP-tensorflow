# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


def corr2d(X, K):
    h, w = K.shape


    if len(X.shape) <= 1:
        X = tf.reshape(X, (X.shape[0], 1))
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1,
                              X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.cast(tf.reduce_sum(X[i: i+h, j: j+w] * K), dtype=tf.float32))

    return Y


def corr2d_multi_in(X, K):
    return tf.reduce_sum([corr2d(X[i], K[i]) for i in range(X.shape[0])], axis=0)


def corr2d_multi_in_out(X, K):
    return tf.stack([corr2d_multi_in(X, k) for k in K], axis=0)


def main():
    X = tf.constant([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                     [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    K = tf.constant([[[0, 1], [2, 3]],
                     [[1, 2], [3, 4]]])
    print(corr2d_multi_in(X, K))


if __name__ == "__main__":
    main()
