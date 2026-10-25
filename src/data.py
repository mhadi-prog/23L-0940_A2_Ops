"""tf.data input pipeline for VisionGuard."""
import random

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
IMG_SIZE = 160
BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# Random augmentation, applied to the training split only
augmenter = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.05, seed=SEED),
    tf.keras.layers.RandomZoom(0.1, seed=SEED),
    tf.keras.layers.RandomContrast(0.1, seed=SEED),
], name="augmentation")


def _load(path, label):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)  # L, P, RGBA -> RGB
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True)    # stretches to a square
    return tf.cast(img, tf.float32), tf.cast(label, tf.float32)         # pixel range stays 0..255


def make_dataset(csv_path, training=False, batch_size=BATCH, shuffle=None, augment=None):
    df = pd.read_csv(csv_path)
    shuffle = training if shuffle is None else shuffle
    augment = training if augment is None else augment

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()                                   # decode and resize once, reuse every epoch
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:                                       # after caching, so each epoch sees new variations
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)