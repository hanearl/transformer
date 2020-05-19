import tensorflow as tf
from tensorflow import keras
import numpy as np


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()

        self.dense_q = keras.layers.Dense(d_model)
        self.dense_k = keras.layers.Dense(d_model)
        self.dense_v = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model

    def scaled_dot_attention(self, query, key, value, mask):
        d_k = key.shape[-1]
        qk = tf.matmul(query, key, transpose_b=True)
        logits = tf.math.divide(qk, d_k ** 0.5)
        if mask is not None:
            logits += (mask * -1e9)

        weights = tf.nn.softmax(logits, axis=-1)
        attention = tf.matmul(weights, value)
        return attention, weights

    def call(self, query, key, value, mask):
        batch_size = query.shape[0]

        query = self.dense_q(query)
        key = self.dense_k(key)
        value = self.dense_v(value)

        mh_query = tf.reshape(query, shape=(batch_size, -1, self.num_heads, self.d_k))
        mh_query = tf.transpose(mh_query, [0, 2, 1, 3])
        mh_key = tf.reshape(key, shape=(batch_size, -1, self.num_heads, self.d_k))
        mh_key = tf.transpose(mh_key, [0, 2, 1, 3])
        mh_value = tf.reshape(value, shape=(batch_size, -1, self.num_heads, self.d_k))
        mh_value = tf.transpose(mh_value, [0, 2, 1, 3])
        attn_value, weights = self.scaled_dot_attention(mh_query, mh_key, mh_value, mask)

        attn_value = tf.transpose(attn_value, [0, 2, 1, 3])
        attn_value = tf.reshape(attn_value, shape=(batch_size, -1, self.d_model))

        attn_value = self.dense(attn_value)
        return attn_value, weights


def test():
    batch = 32
    seq_len = 200
    d_model = 512

    random_init = tf.random_normal_initializer()
    query = tf.Variable(initial_value=random_init(shape=(batch, seq_len, d_model)))
    key = tf.Variable(initial_value=random_init(shape=(batch, seq_len, d_model)))
    value = tf.Variable(initial_value=random_init(shape=(batch, seq_len, d_model)))

    zero_init = tf.zeros_initializer()
    one_init = tf.ones_initializer()
    zero = tf.Variable(initial_value=zero_init(shape=(batch, seq_len)))
    one = tf.Variable(initial_value=one_init(shape=(batch, seq_len)))

    mha = MultiHeadAttention(8, 512)
    att = mha(query, key, value, zero[:, tf.newaxis, tf.newaxis, :])
    print(att)

test()
