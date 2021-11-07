# -*- coding: utf-8 -*-


import sys
sys.path.append('.')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import data as tfdata
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import initializers as init
from tensorflow import optimizers
from tensorflow import losses


def main():
    num_epochs = 5
    batch_size = 256
    lr = 0.01

    # 数据加载
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # 归一化
    x_train = tf.cast(x_train, tf.float32) / 255
    x_test = tf.cast(x_test, tf.float32) / 255

    # 模型定义
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(10, activation=tf.nn.softmax)
    ])


    model.compile(optimizer=optimizers.SGD(0.1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=256)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(test_accuracy)


if __name__ == "__main__":
    main()
