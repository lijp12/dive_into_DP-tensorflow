# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow import initializers as init


def get_net():
    net = keras.models.Sequential()
    net.add(keras.layers.Dense(1))
    return net


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j+1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = tf.concat([X_train, X_part], axis=0)
            y_train = tf.concat([y_train, y_part], axis=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        net.compile(loss=tf.keras.losses.mean_squared_logarithmic_error,
                    optimizer=tf.keras.optimizers.Adam(learning_rate))
        history = net.fit(data[0], data[1], validation_data=(data[2], data[3]),
                          epochs=num_epochs, batch_size=batch_size,
                          validation_freq=1, verbose=0)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        print('fold %d, train rmse %f, valid rmse %f' % (i, loss[-1], val_loss[-1]))

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='train')
    plt.plot(val_loss, label='valid')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def main():
    # 数据初识
    train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
    # print(train_data.shape)
    # print(test_data.shape)
    # print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    # 预处理1：标准化
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    all_features = all_features.fillna(0)

    # 预处理2：离散值转化为指示特征
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)

    # 转化成模型数据
    n_train = train_data.shape[0]
    train_features = np.array(all_features[:n_train].values, dtype=np.float)
    test_features = np.array(all_features[n_train:].values, dtype=np.float)
    train_labels = np.array(train_data.SalePrice.values.reshape(-1, 1), dtype=np.float)

    # # 模型选择
    # k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    # k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)


    # 在完整数据集上训练并预测
    x_train = tf.convert_to_tensor(train_features, dtype=tf.float32)
    y_train = tf.convert_to_tensor(train_labels, dtype=tf.float32)
    x_test = tf.convert_to_tensor(test_features, dtype=tf.float32)

    model = keras.models.Sequential([
        keras.layers.Dense(1)
    ])
    adam = keras.optimizers.Adam(0.5)
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.mean_squared_logarithmic_error)
    model.fit(x_train, y_train, epochs=200, batch_size=32, verbose=0)
    pred = np.array(model.predict(x_test))
    test_data['SalePrice'] = pd.Series(pred.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
