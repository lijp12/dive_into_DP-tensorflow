# -*- coding: utf-8 -*-


import tensorflow as tf


def main():
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    cpus = tf.config.experimental.list_physical_devices(device_type="CPU")
    print(gpus)
    print(cpus)

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


    # 指定设备:tf.device()
    with tf.device('/GPU:0'):
        a = tf.constant([1, 2, 3], dtype=tf.float32)
        b = tf.random.uniform((3,))

        print(tf.exp(a + b) * 2)


if __name__ == "__main__":
    main()
