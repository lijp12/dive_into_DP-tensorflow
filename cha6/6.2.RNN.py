# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


def main():
    X, W_xh = tf.random.normal(shape=(3, 1)), tf.random.normal(shape=(1, 4))
    H, W_hh = tf.random.normal(shape=(3, 4)), tf.random.normal(shape=(4, 4))

    print(tf.matmul(X, W_xh) + tf.matmul(H, W_hh))

    print(tf.matmul(tf.concat([X, H], axis=-1), tf.concat([W_xh, W_hh], axis=0)))


if __name__ == "__main__":
    main()
