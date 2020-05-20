import tensorflow as tf
from tensorflow.keras import layers


class PointWiseFeedForwardNetwork(layers.Layer):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.pw_ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, inputs, **kwargs):
        return self.pw_ffn(inputs)