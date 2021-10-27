# -*- coding: utf-8 -*-


import tensorflow as tf
import random


def gen_data():
    num_inputs = 2
    num_examples = 1000
    true_w = [-2, 4.5]
    true_b = 3.2
    features = tf.random.normal((num_examples, num_inputs), stddev=1)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += tf.random.normal(labels.shape, stddev=0.01)
    return features, labels

def data_iter(features, labels, batch_size):
    num_examples = features.shape[0]
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i:min(num_examples, i+batch_size)]
        yield tf.gather(features, axis=0, indices=j), tf.gather(labels, axis=0, indices=j)

def linreg(X, w, b):
    return tf.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) **2 / 2

def sgd(params, lr, batch_size, grads):
    for i, param in enumerate(params):
        param.assign_sub(lr * grads[i] / batch_size)

def main():
    # tf.enable_eager_execution()

    num_inputs = 2
    num_epochs = 3
    batch_size = 10
    lr = 0.03
    net = linreg
    loss = squared_loss

    w = tf.Variable(tf.random.normal((num_inputs, 1), stddev=0.01))
    b = tf.Variable(tf.zeros((1,)))

    features, labels = gen_data()
    for epoch in range(num_epochs):
        for X, y in data_iter(features, labels, batch_size):
            with tf.GradientTape() as t:
                t.watch([w, b])
                l = tf.reduce_sum(loss(net(X, w, b), y))
            grads = t.gradient(l, [w, b])
            sgd([w, b], lr, batch_size, grads)
        train_l = loss(net(features, w, b), labels)
        print("epoch %d, loss %f" % (epoch + 1, tf.reduce_mean(train_l)))


if __name__ == "__main__":
    main()
