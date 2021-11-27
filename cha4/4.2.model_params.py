# -*- coding: utf-8 -*-


import tensorflow as tf


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(
            units=10,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
            bias_initializer=tf.zeros_initializer()
        )

        self.d2 = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.ones_initializer(),
            bias_initializer=tf.ones_initializer()
        )

    def call(self, inputs, training=None, mask=None):
        output = self.d1(inputs)
        output = self.d2(output)
        return output


def my_init():
    return tf.keras.initializers.Ones()


def main():
    # net = tf.keras.models.Sequential()
    # net.add(tf.keras.layers.Flatten())
    # net.add(tf.keras.layers.Dense(units=256, activation=tf.nn.relu))
    # net.add(tf.keras.layers.Dense(units=10))

    X = tf.random.uniform((2, 20))
    # Y = net(X)
    # print(Y)
    #
    # for weight in net.weights:
    #     print(weight, type(weight))

    # net = Linear()
    # print(net(X))
    #
    # print(net.get_weights())


    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, kernel_initializer=my_init()))

    Y = model(X)
    print(model.weights)



if __name__ == "__main__":
    main()
