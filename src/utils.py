"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
from math import ceil

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle


def get_num_classes(data_path):
    return len(pd.read_csv(data_path, header=None, usecols=[0])[0].unique())


def create_dataset(data_path, alphabet="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""",
                   max_length=1014, batch_size=128, is_training=True):
    label_converter = lambda x: int(x) - 1
    data = pd.read_csv(data_path, header=None, converters={0: label_converter})
    num_iters = ceil(data.shape[0] / batch_size)
    if is_training:
        data = shuffle(data, random_state=42)
    num_columns = data.shape[1]
    for idx in range(2, num_columns):
        data[1] += data[idx]
    data = data.drop([idx for idx in range(2, num_columns)], axis=1).values
    alphabet = list(alphabet)
    identity_mat = np.identity(len(alphabet))

    def generator():
        for row in data:
            label, text = row
            text = np.array([identity_mat[alphabet.index(i)] for i in list(str(text)) if i in alphabet], dtype=np.float32)
            if len(text) > max_length:
                text = text[:max_length]
            elif 0 < len(text) < max_length:
                text = np.concatenate((text, np.zeros((max_length - len(text), len(alphabet)), dtype=np.float32)))
            elif len(text) == 0:
                text = np.zeros((max_length, len(alphabet)), dtype=np.float32)
            yield text.T, label

    return tf.data.Dataset.from_generator(generator, (tf.float32, tf.int32),
                                          ((len(alphabet), max_length), (None))).batch(batch_size), num_iters
