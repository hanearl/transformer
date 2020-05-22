import tensorflow as tf
from tensorflow import keras
import numpy as np

from model.ffnn import PointWiseFeedForwardNetwork
from model.attention import MultiHeadAttention
from model.embedding import positional_encoding


class EncoderLayer(keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        mha_params = {"num_heads": num_heads, "d_model": d_model}
        self.mha = MultiHeadAttention(**mha_params)

        ffn_params = {"dff": dff, "d_model": d_model}
        self.ffn = PointWiseFeedForwardNetwork(**ffn_params)

        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training):
        attn_out, _ = self.mha(x, x, x, mask)
        attn_out = self.dropout1(attn_out, training=training)
        attn_out = self.layer_norm1(attn_out + x)

        output = self.ffn(attn_out)
        output = self.dropout2(output, training=training)
        output = self.layer_norm1(output + attn_out)

        return output


class Encoder(keras.layers.Layer):
    def __init__(self, num_encoder, num_heads, d_model, dff, max_seq_len, vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model

        params = {"num_heads": num_heads, "d_model": d_model, "dff": dff}
        self.encoder_stack = [EncoderLayer(**params) for _ in range(num_encoder)]

        self.pos_encoding = positional_encoding(d_model, max_seq_len)

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask, training):
        seq_len = x.shape[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for enc in self.encoder_stack:
            x = enc(x, mask, training)

        return x


def test():
    encoder = Encoder(3, 8, 512, 2048, 64, 200)
    temp_input = tf.random.uniform((64, 64), dtype=tf.int64, minval=0, maxval=200)
    mask = tf.Variable(initial_value=tf.zeros_initializer()(shape=(64, 64)))
    print(encoder(temp_input, mask[:, np.newaxis, np.newaxis, :], True))

# test()