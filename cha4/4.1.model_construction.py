# -*- coding: utf-8 -*-



import tensorflow as tf


# tf.keras.Model类的__init__函数和call函数，分别用于创建模型参数和定义前向计算
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        output = self.dense2(x)
        return output


class FancyMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(units=20, activation=tf.nn.relu)


    def call(self, inputs, training=None, mask=None):
        x = self.flatten(inputs)
        x = tf.nn.relu(tf.matmul(x, self.rand_weight) + 1)
        x = self.dense(x)

        while tf.norm(x) > 1:
            x /= 2
        if tf.norm(x) < 0.8:
            x *= 10
        return tf.reduce_sum(x)


def main():
    X = tf.random.uniform((2, 20))
    net = MLP()
    print(net(X))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(10),
    ])
    print(model(X))

    fancy_net = FancyMLP()
    print(fancy_net(X))


if __name__ == "__main__":
    main()
