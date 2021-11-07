# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np


def relu(x):
    return tf.math.maximum(x, 0)


def net(X, W1, B1, W2, B2):
    X = tf.reshape(X, shape=[-1, W1.shape[0]])
    h = relu(tf.matmul(X, W1) +B1)
    return tf.math.softmax(tf.matmul(h, W2) + B2)


def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(y, y_hat)


def evaluate_accuracy(iter, model, W1, B1, W2, B2):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(iter):
        y = tf.cast(y, dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(model(X, W1, B1, W2, B2), axis=-1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n


def train_process(model, train_iter, test_iter, loss, num_epoches, batch_size, params, lr, trainer=None):
    for epoch in range(1, num_epoches + 1):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, Y in train_iter:
            with tf.GradientTape() as t:
                y_hat = model(X, params[0], params[1], params[2], params[3])
                l = tf.reduce_sum(loss(y_hat, Y))
            grads = t.gradient(l, params)

            if trainer is None:
                for i, param in enumerate(params):
                    param.assign_sub(grads[i] * lr / batch_size)
            else:
                trainer.apply_gradients(zip([grad/batch_size for grad in grads], params))

            Y = tf.cast(Y, tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=-1) == tf.cast(Y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += Y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net, params[0], params[1], params[2], params[3])
        print(
            "epoch %d, loss %.4f train acc %.3f test acc %.3f" % (epoch, train_l_sum / n, train_acc_sum / n, test_acc))


def main():
    batch_size = 256
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = tf.cast(x_train, tf.float32) / 255
    x_test = tf.cast(x_test, tf.float32) / 255

    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # 参数定义
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    W1 = tf.Variable(tf.random.normal(shape=[num_inputs, num_hiddens], mean=0, stddev=0.01), dtype=tf.float32)
    B1 = tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32))
    W2 = tf.Variable(tf.random.normal(shape=[num_hiddens, num_outputs], mean=0, stddev=0.01), dtype=tf.float32)
    B2 = tf.Variable(tf.random.normal(shape=[num_outputs], stddev=0.1), dtype=tf.float32)

    num_epochs, lr = 5, 0.5

    params = [W1, B1, W2, B2]

    train_process(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr=0.1)


if __name__ == "__main__":
    main()
