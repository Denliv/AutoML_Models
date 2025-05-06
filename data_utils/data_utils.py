import os
import numpy as np
import random

from enum import IntEnum
from sklearn.preprocessing import LabelEncoder

class DataEncoder:
    def __init__(self, categories):
        self.encoder = LabelEncoder().fit(categories)

    def to_encoded(self, data):
        return self.encoder.transform(data)

    def to_decoded(self, data):
        return self.encoder.inverse_transform(data)

class Data(IntEnum):
    PyTorch = 1
    Keras = 3

def count_classes_volume(classes, data_path):
    counts = []
    join = os.path.join
    listdir = os.listdir

    for c in classes:
        dir_path = join(data_path, c)
        counts.append((c, len(listdir(dir_path))))
    return counts

def print_classes_volume(classes, data_path):
    join = os.path.join
    listdir = os.listdir

    for c in classes:
        print(f"{c} | {len(listdir(join(data_path, c)))} files")

def add_color_channel(image_data, axis):
    return np.expand_dims(image_data, axis=axis)

def to_categorical(data, num_classes):
    return np.eye(num_classes, dtype='uint8')[data]

def normalize(data):
    return data.astype("float32") / 255.0