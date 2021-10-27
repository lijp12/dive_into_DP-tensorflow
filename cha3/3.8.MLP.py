# -*- coding: utf-8 -*-


import tensorflow as tf
from matplotlib import pyplot  as plt
import numpy as np


def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize


def xyplot(x, y, name):
    set_figsize(figsize=(5, 2.5))
    plt.plot(x.numpy(), y.numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()


def main():
    x = tf.Variable(tf.range(-8, 8, 0.1), dtype=tf.float32)

    # relu
    # y = tf.nn.relu(x)
    # # xyplot(x, y, 'relu')
    #
    # with tf.GradientTape() as t:
    #     t.watch(x)
    #     y = tf.nn.relu(x)
    # dy_dx = t.gradient(y, x)
    # xyplot(x, dy_dx, 'grad of relu')


    # # sigmoid
    # y = tf.nn.sigmoid(x)
    # # xyplot(x, y, 'sigmoid')
    # with tf.GradientTape() as t:
    #     t.watch(x)
    #     y = tf.nn.sigmoid(x)
    # dy_dx = t.gradient(y, x)
    # xyplot(x, dy_dx, 'grad of sigmoid')


    # tanh
    y = tf.nn.tanh(x)
    # xyplot(x, y, 'tanh')
    with tf.GradientTape() as t:
        t.watch(x)
        y = tf.nn.tanh(x)
    dy_dx = t.gradient(y, x)
    xyplot(x, dy_dx, 'grad of tanh')


if __name__ == "__main__":
    main()
