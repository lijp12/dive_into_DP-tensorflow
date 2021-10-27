# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def main():
    batch_size = 256
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = tf.cast(x_train, tf.float32) / 255
    x_test = tf.cast(x_test, tf.float32) / 255

    x_train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(batch_size)
    x_test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # 参数定义
    num_inputs, num_outputs, num_hiddens = 768, 10, 256
    W1 = tf.Variable(tf.random.normal(shape=[num_inputs, num_hiddens], mean=0, stddev=0.01), dtype=tf.float32)
    B1 = tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32))
    W2 = tf.Variable(tf.random.normal(shape=[num_hiddens, num_outputs], mean=0, stddev=0.01), dtype=tf.float32)
    B2 = tf.Variable(tf.random.normal(shape=[num_outputs], stddev=0.1), dtype=tf.float32)





if __name__ == "__main__":
    main()
