import tensorflow as tf
from tensorflow import keras
import numpy as np


class PositionalEncodingLayer(keras.layers.Layer):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncodingLayer, self).__init__()
        self.pos_encoding = self.positional_encoding(d_model, max_seq_len)

    def positional_encoding(self, d_model, max_seq_len):
        angle = np.arange(d_model)
        angle = 1 / np.power(10000, (2 * (angle // 2)) / d_model)
        pos_encoding = np.arange(max_seq_len)[:, np.newaxis] * angle[np.newaxis, :]
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        return tf.cast(pos_encoding, dtype=tf.float32)

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        return tf.math.add(inputs, self.pos_encoding)


def test():
    d_model = 512
    max_len_seq = 300
    batch_size = 32

    positional_encoding_layer = PositionalEncodingLayer(d_model, max_len_seq)
    inputs = tf.Variable(initial_value=tf.zeros_initializer()(shape=(batch_size, max_len_seq, d_model)))

    print(positional_encoding_layer(inputs))

test()