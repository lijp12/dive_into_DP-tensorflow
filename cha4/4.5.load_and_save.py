# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


def main():
    # load and save NDarray
    # x = tf.ones(3)
    # print(x)
    # np.save('x.npy', x)
    # x2 = np.load('x.npy')
    # print(x2)
    #
    # y = tf.zeros(4)
    # np.save('xy.npy', [x, y])
    # x2, y2 = np.load('xy.npy', allow_pickle=True)
    # print(x2, y2)
    #
    # mydict = {
    #     'x': x,
    #     'y': y
    # }
    # np.save('mydict.npy', mydict)
    # mydict2 = np.load('mydict.npy', allow_pickle=True)
    # print(mydict2)

    # load and save model parameters
    X = tf.random.uniform((2, 20))
    print(X)

    net = MLP()
    Y = net(X)
    print(Y)

    net.save_weights('4.5saved_model.h5')

    net2 = MLP()
    print(net2(X))
    net2.load_weights('4.5saved_model.h5')
    Y2 = net2(X)
    print(Y2 == Y)


if __name__ == "__main__":
    main()
