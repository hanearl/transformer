import tensorflow_datasets as tfds
import tensorflow as tf

"""
tensorflow tutorial 참고
https://www.tensorflow.org/tutorials/text/transformer
"""


class HelperTFDS:
    def __init__(self, batch_size, buffer_size, max_length):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.load()

    def load(self):
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       data_dir="tfds",
                                       with_info=True,
                                       as_supervised=True)
        self.train_examples, self.val_examples = examples['train'], examples['validation']

        self.tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in self.train_examples), target_vocab_size=2**13)
        self.tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in self.train_examples), target_vocab_size=2**13)

    def encode(self, lang1, lang2):
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            lang1.numpy()) + [self.tokenizer_pt.vocab_size + 1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            lang2.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return lang1, lang2

    def tf_encode(self, pt, en):
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    def filter_max_length(self, x, y):
      return tf.logical_and(tf.size(x) <= self.max_length,
                            tf.size(y) <= self.max_length)

    def get_dataset(self):
        train_dataset = self.train_examples.map(self.tf_encode)
        train_dataset = train_dataset.filter(self.filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset\
                            .shuffle(self.buffer_size)\
                            .padded_batch(self.batch_size, padded_shapes=([None], [None]))\
                            .prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = self.val_examples.map(self.tf_encode)
        val_dataset = val_dataset\
                            .filter(self.filter_max_length)\
                            .padded_batch(self.batch_size, padded_shapes=([None], [None]))

        return train_dataset, val_dataset


# def test():
#     tfds_helper = HelperTFDS(64, 20000, 40)
#     train_dataset, val_dataset = tfds_helper.get_dataset()
#     print(train_dataset)
#
# test()
