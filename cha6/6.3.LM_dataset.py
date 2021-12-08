# -*- coding: utf-8 -*-


import numpy as np
import random


# 时序数据采样之随机采样
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos:pos+num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i+batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield np.array(X, ctx), np.array(Y, ctx)


# 时序数据采样之相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i+num_steps]
        Y = indices[:, i+1: i+num_steps+1]
        yield X, Y


def main():
    # f = open('./jaychou_lyrics.txt', 'r')
    # corpus_chars = f.read()
    # f.close()
    #
    # corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    #
    # # 建立字符索引
    # idx_to_char = list(set(corpus_chars))
    # char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    # vocab_size = len(char_to_idx)
    # print('vocab size:\t', vocab_size)
    #
    # # 转化训练数据
    # corpus_indices = [char_to_idx[char] for char in corpus_chars]
    # sample = corpus_indices[:20]
    # print('chars:\t', ''.join([idx_to_char[idx] for idx in sample]))
    # print('indices:\t', sample)
    my_seq = list(range(30))
    for X, Y in data_iter_consecutive(my_seq, 2, 6):
        print(X, Y)


if __name__ == "__main__":
    main()
