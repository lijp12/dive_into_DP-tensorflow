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
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    # 数据迭代
    train_iter = tfdata.Dataset.from_tensor_slices((x_train, y_train)).\
        shuffle(buffer_size=x_train.shape[0]).batch(batch_size)

    # 模型定义
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(10, activation=tf.nn.softmax)
    ])




if __name__ == "__main__":
    main()
