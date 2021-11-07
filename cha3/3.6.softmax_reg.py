# -*- coding: utf-8 -*-


import sys
sys.path.append('.')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import fashion_mnist_display


# 模型构造
def softmax(logits, axis=-1):
    return tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)


def net(X, W, b):
    logits = tf.matmul(tf.reshape(X, shape=(-1, W.shape[0])), W) + b
    return softmax(logits)


def cross_entropy(y_hat, y):
    y = tf.cast(tf.reshape(y, shape=[-1, 1]), dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]), dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat, y)+1e-8)


def accuracy(y_hat, y):
    return np.mean(tf.argmax(y_hat, axis=-1) == y)


def evaluate_accuracy(data_iter, net, W, b):
    acc_sum, n = 0., 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y, dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X, W, b), axis=-1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n


def train_ch3(num_epochs, train_iter, test_iter, net, params, trainer, lr, batch_size):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X, params[0], params[1])
                l = tf.reduce_sum(cross_entropy(y_hat, y))
            grads = tape.gradient(l, params)
            if trainer is None:
                for i, param in enumerate(params):
                    param.assign_sub(lr * grads[i] / batch_size)
            else:
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))

            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=-1) == tf.cast(y, dtype=tf.int64),
                                                   dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net, params[0], params[1])
        print("epoch %d, loss %.4f train acc %.3f test acc %.3f" % (epoch, train_l_sum / n, train_acc_sum / n, test_acc))


def main():
    # 数据加载
    batch_size = 256
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = tf.cast(x_train, tf.float32) / 255
    x_test = tf.cast(x_test, tf.float32) / 255
    train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # 模型参数初始化
    num_inputs = 784
    num_outputs = 10
    W = tf.Variable(tf.random.normal((num_inputs, num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
    b = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))

    # 模型训练
    num_epochs, lr = 5, 0.1
    trainer = tf.keras.optimizers.SGD(lr)
    train_ch3(num_epochs, train_iter, test_iter, net, [W, b], trainer, lr, batch_size)

    # 模型infer
    X, y = iter(test_iter).next()
    true_labels = fashion_mnist_display.get_fashion_mnist_labels(y.numpy())
    pred_labels = fashion_mnist_display.get_fashion_mnist_labels(tf.argmax(net(X, W, b), axis=-1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
    fashion_mnist_display.show_fashion_mnist(X[0:9], titles[0:9])


if __name__ == "__main__":
    main()
