# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.datasets import fashion_mnist


def main():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = tf.cast(x_train, dtype=tf.float32) / 255.0
    x_test = tf.cast(x_test, dtype=tf.float32) /255.0

    model = keras.models.Sequential([
        layers.Flatten(input_shape=[28, 28]),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=256, validation_data=(x_test, y_test), validation_freq=1)

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test Acc:", test_acc)


if __name__ == "__main__":
    main()
