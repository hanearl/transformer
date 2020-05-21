import tensorflow as tf
from tensorflow import keras

from encoder import Encoder
from decoder import Decoder


class Transformer(keras.Model):
    def __init__(self, num_encoder, num_decoder, d_model,
                 num_heads, dff, seq_len, vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        encoder_params = {"num_encoder": num_encoder, "num_heads": num_heads, "d_model": d_model,
                          "dff": dff, "max_seq_len": seq_len, "vocab_size": vocab_size, "rate": rate}
        self.encoder = Encoder(**encoder_params)
        decoder_params = {"num_decoder": num_decoder, "num_heads": num_heads, "d_model": d_model,
                          "dff": dff, "max_seq_len": seq_len, "vocab_size": vocab_size, "rate": rate}
        self.decoder = Decoder(**decoder_params)

        self.final_dense = keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inp, tar, training, enc_mask, look_ahead_mask,
             dec_mask):
        enc_out = self.encoder(inp, enc_mask, training)
        dec_out = self.decoder(tar, enc_out, look_ahead_mask, dec_mask, training)

        output = self.final_dense(dec_out)
        return output


def test():
    # TODO
    # seq_len이 다른문장이 와도 학습 가능하도록 변경하기
    # 첫 embedding 할 때 scaling 추가

    transformer = Transformer(num_encoder=2, num_decoder=2, d_model=512,
                 num_heads=4, dff=1024, seq_len=200, vocab_size=200)

    temp_input = tf.random.uniform((64, 200), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 200), dtype=tf.int64, minval=0, maxval=200)
    print(transformer(temp_input, temp_target, False, None, None, None))

# test()