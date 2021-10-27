# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow import data as tfdata
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init
from tensorflow import losses
from tensorflow import optimizers


def main():
    num_inputs = 2
    num_examples = 1000

    # 生成数据
    true_w = [1, 2]
    true_b = 3
    features = tf.random.normal((num_examples, num_inputs), stddev=1)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += tf.random.normal(labels.shape, stddev=0.01)

    # 迭代生成训练数据
    batch_size = 10
    dataset = tfdata.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=num_examples)
    dataset = dataset.batch(batch_size)
    data_iter = iter(dataset)

    # 定义模型
    model = keras.Sequential()
    model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01)))

    # 定义损失函数
    loss = losses.MeanSquaredError()

    # 定义优化算法
    trainer = optimizers.SGD(learning_rate=0.03)

    # 训练模型
    num_epochs = 3
    for epoch in range(num_epochs):
        for batch, (X, y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                l = loss(model(X, training=True), y)
            grads = tape.gradient(l, model.trainable_variables)
            trainer.apply_gradients(zip(grads, model.trainable_variables))
        l = loss(model(features), labels)
        print('epoch %d, loss %f' % (epoch + 1, l))

    print(model.get_weights())


if __name__ == "__main__":
    main()
