from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def _split_data(x, y):
    return train_test_split(x, y, train_size=0.7, random_state=1)


def load_sk_digits():
    return _split_data(*load_digits(return_X_y=True))


def load_sk_digits_summarized():
    x, y = load_digits(return_X_y=True)

    x = np.where(x < 5, 0, x)
    x = np.where(x > 10, 2, x)
    x = np.where(x > 2, 1, x)

    return _split_data(x, y)


def load_light_digits():
    digit_path = Path('MNIST_Light')
    target_dirs = [d for d in digit_path.iterdir() if d.is_dir()]

    x = np.array([np.array(Image.open(f)) for d in target_dirs
                  for f in d.glob('*.png')]).reshape(5000, 400) / 255
    y = np.array([int(target.name) for target in target_dirs for i in range(500)])

    return _split_data(x, y)
