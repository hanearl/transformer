import tensorflow as tf
import numpy as np


def positional_encoding(d_model, max_seq_len):
    angle = np.arange(d_model)
    angle = 1 / np.power(10000, (2 * (angle // 2)) / d_model)
    pos_encoding = np.arange(max_seq_len)[:, np.newaxis] * angle[np.newaxis, :]
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    return tf.cast(pos_encoding, dtype=tf.float32)


def test():
    d_model = 512
    max_len_seq = 300
    batch_size = 32

    pos_encoding = positional_encoding(d_model, max_len_seq)
    inputs = tf.Variable(initial_value=tf.zeros_initializer()(shape=(batch_size, max_len_seq, d_model)))

    print(pos_encoding(inputs))

# test()