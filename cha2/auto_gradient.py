# -*- coding: utf-8 -*-


import tensorflow as tf


def main():
    x = tf.reshape(tf.Variable(range(4), dtype=tf.float32), (4, 1))
    print(x)

    with tf.GradientTape() as t:
        t.watch(x)
        y = 2 * tf.matmul(tf.transpose(x), x)
    dy_dx = t.gradient(y, x)
    print(dy_dx)


if __name__ == "__main__":
    main()
