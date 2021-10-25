# -*- coding: utf-8 -*-


import tensorflow as tf


def main():
    # # tensor
    x = tf.constant(range(12))
    # print(x)
    # print(x.shape)
    # print(len(x))
    #
    X = tf.reshape(x, (3, 4))
    print(X)
    #
    # print(tf.zeros((2, 3, 4)))
    # print(tf.ones([3, 4]))
    #
    # Y = tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    # print(Y)
    #
    # print(tf.random.normal(shape=[3, 4], mean=0, stddev=1))
    #
    #
    # # op
    # print(X + Y)
    # print(X * Y)
    # print(X / Y)
    #
    # Y = tf.cast(Y, dtype=tf.float32)
    # print(tf.exp(Y))
    #
    # Y = tf.cast(Y, dtype=tf.int32)
    # print(tf.matmul(X, tf.transpose(Y)))
    #
    # print(tf.concat([X, Y], axis=0))
    # print(tf.concat([X, Y], axis=1))
    # print(tf.equal(X, Y))
    # print(tf.reduce_sum(X))
    #
    # X = tf.cast(X, dtype=tf.float32)
    # print(tf.norm(X))


    # # broadcast
    # A = tf.reshape(tf.constant(range(3)), (3, 1))
    # B = tf.reshape(tf.constant(range(2)), (1, 2))
    # print(A + B)

    # # index
    # print(X[1:3])
    # X = tf.Variable(X)
    # X[1, 2].assign(9)
    # print(X)
    #
    # X[1:2, :].assign(tf.ones(X[1:2, :].shape, dtype=tf.int32) * 12)
    # print(X)

    # tensor和numpy互换
    import numpy as np
    P = np.ones((2, 3))
    D = tf.constant(P)

    print(np.array(D))


if __name__ == "__main__":
    main()
