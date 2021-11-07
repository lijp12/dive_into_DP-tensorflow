# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np


def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1

    keep_prob = 1 - drop_prob

    if keep_prob == 0:
        return tf.zeros_like(X)

    mask = tf.random.uniform(shape=X.shape, minval=0, maxval=1) < keep_prob
    return tf.cast(mask, dtype=tf.float32) * tf.cast(X, dtype=tf.float32) / keep_prob


def main():
    # X = tf.reshape(tf.range(0, 16), shape=(2, 8))
    # print(dropout(X, 0))
    # print(dropout(X, 0.5))
    # print(dropout(X, 1.0))

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

    W1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens1), stddev=0.01))
    B1 = tf.Variable(tf.zeros(num_hiddens1))

    W2 = tf.Variable(tf.random.normal(shape=(num_hiddens1, num_hiddens2), stddev=0.1))
    B2 = tf.Variable(tf.zeros(num_hiddens2))

    W3 = tf.Variable(tf.random.truncated_normal(shape=(num_hiddens2, num_outputs), stddev=0.01))
    B3 = tf.Variable(tf.zeros(num_outputs))

    params = [W1, B1, W2, B2, W3, B3]

    drop_prob1, drop_prob2 = 0.2, 0.5

    def net(X, is_training=False):
        X = tf.reshape(X, shape=(-1, num_inputs))
        H1 = tf.nn.relu(tf.matmul(X, W1,) + B1)
        if is_training:
            H1 = dropout(H1, drop_prob1)
        H2 = tf.nn.relu(tf.matmul(H1, W2) + B2)
        if is_training:
            H2 = dropout(H2, drop_prob2)
        return tf.nn.softmax(tf.matmul(H2, W3) + B3)

    def evaluate_accuracy(iter, net):
        acc_sum ,n = 0.0, 0
        for _, (X, y) in enumerate(iter):
            y = tf.cast(y, dtype=tf.int64)
            acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64)==y)
            n += y.shape[0]
        return acc_sum / n

    def train_process(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
        for epoch in range(num_epochs):
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in train_iter:
                with tf.GradientTape() as tape:
                    y_hat = net(X, is_training=True)
                    l = loss(y_hat, tf.one_hot(y, depth=10, axis=-1, dtype=tf.float32))
                grads = tape.gradient(l, params)

                if trainer is None:
                    params[0].assign_sub(grads[0] * lr)
                    params[1].assign_sub(grads[1] * lr)
                else:
                    trainer.apply_gradients(zip(grads, params))

                y = tf.cast(y, dtype=tf.float32)
                train_l_sum += l.numpy()
                train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=-1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
                n += y.shape[0]

            test_acc = evaluate_accuracy(test_iter, net)
            print('epoch %d, loss %.4f, train_acc %.3f, test acc %.3f' % (epoch+1, train_l_sum / n, train_acc_sum / n, test_acc))

    loss = tf.losses.CategoricalCrossentropy()
    num_epochs, lr, batch_size = 5, 0.5, 256
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = tf.cast(x_train, dtype=tf.float32) / 255
    x_test = tf.cast(x_test, dtype=tf.float32) / 255
    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    train_process(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)




if __name__ == "__main__":
    main()