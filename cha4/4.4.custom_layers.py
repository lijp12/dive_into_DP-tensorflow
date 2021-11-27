# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np


#自定义layer
class CenterenLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)


class myDence(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=[input_shape[-1], self.units],
            initializer=tf.random_normal_initializer()
        )
        self.b = self.add_weight(
            name='b',
            shape=[self.units],
            initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


def main():
    X = tf.random.uniform((2, 20))

    # layer = CenterenLayer()
    # print(layer(np.array([1, 2, 3, 4, 5])))
    #

    # model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(20))
    # model.add(CenterenLayer())
    #
    # Y = model(X)
    # print(Y)
    # print(tf.reduce_mean(Y))

    # dense = myDence(3)
    # print(dense(X))
    # print(dense.get_weights())
    # print(dense.weights)

    net = tf.keras.models.Sequential()
    net.add(myDence(8))
    net.add(myDence(1))
    print(net(X))


if __name__ == "__main__":
    main()
