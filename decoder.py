import tensorflow as tf
import numpy as np
from tensorflow import keras

from embedding import positional_encoding
from attention import MultiHeadAttention
from ffnn import PointWiseFeedForwardNetwork


class DecoderLayer(keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        mha_param = {"num_heads": num_heads, "d_model": d_model}
        self.masked_mha = MultiHeadAttention(**mha_param)
        self.mha = MultiHeadAttention(**mha_param)

        ffn_params = {"dff": dff, "d_model": d_model}
        self.ffn = PointWiseFeedForwardNetwork(**ffn_params)

        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()
        self.layer_norm3 = keras.layers.LayerNormalization()

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, x, enc_output, lookahead_mask, padding_mask, training):
        attn_out1, _ = self.masked_mha(x, x, x, lookahead_mask)
        attn_out1 = self.dropout1(attn_out1, training=training)
        attn_out1 = self.layer_norm1(attn_out1 + x)


        attn_out2, _ = self.mha(attn_out1, enc_output, enc_output, padding_mask)
        attn_out2 = self.dropout2(attn_out2, training=training)
        attn_out2 = self.layer_norm2(attn_out2 + attn_out1)

        output = self.ffn(attn_out2)
        output = self.dropout3(output, training=training)
        output = self.layer_norm3(output + attn_out2)

        return output


class Decoder(keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_seq_len, num_decoder, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model

        self.embedding = keras.layers.Embedding(vocab_size, d_model)

        self.pos_encoding = positional_encoding(d_model, max_seq_len)

        params = {"num_heads": num_heads, "d_model": d_model, "dff": dff}
        self.decoder_stack = [DecoderLayer(**params) for _ in range(num_decoder)]

        self.dropout = keras.layers.Dropout(rate)

    def call(self, x, enc_output, lookahead_mask, mask, training):
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding
        x = self.dropout(x, training=training)

        for i, dec in enumerate(self.decoder_stack):
            x = dec(x, enc_output, lookahead_mask, mask, training)

        return x


def test():
    def create_look_ahead_mask(size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    # vocab_size, d_model, max_seq_len, num_decoder, num_heads, dff,
    decoder = Decoder(200, 512, 128, 4, 8, 2048)
    temp_input = tf.random.uniform((32, 128), dtype=tf.int64, minval=0, maxval=200)
    enc_input = tf.random.uniform((32, 128, 512), dtype=tf.float32, minval=0, maxval=1)
    mask = tf.Variable(initial_value=tf.zeros_initializer()(shape=(32, 128)))
    lookahead_mask = create_look_ahead_mask(temp_input.shape[1])
    print(decoder(temp_input, enc_input, lookahead_mask, mask[:, np.newaxis, np.newaxis, :], True))

# test()