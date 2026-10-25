"""Model definitions for VisionGuard."""
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 160


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def build_baseline_cnn(img_size=IMG_SIZE, dropout=0.4, lr=1e-3):
    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)
    for i, filters in enumerate([32, 64, 128], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = models.Model(inputs, outputs, name="baseline_cnn")
    compile_model(model, lr)
    return model


def build_transfer(img_size=IMG_SIZE, dropout=0.3, lr=1e-3):
    """MobileNetV2 (ImageNet weights, no classifier), frozen, plus our own head."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3), name="image")
    # MobileNetV2 was trained on pixels scaled to [-1, 1]; this is exactly
    # what keras.applications.mobilenet_v2.preprocess_input does.
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
    x = base(x, training=False)          # keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = models.Model(inputs, outputs, name="mobilenetv2_transfer")
    compile_model(model, lr)
    return model, base


def unfreeze_last_block(model, base, lr=1e-5, prefixes=("block_16", "Conv_1")):
    """Stage 2: unfreeze only the last block (block_16 + final 1x1 conv).
    BatchNorm layers stay frozen so their statistics aren't disturbed."""
    base.trainable = True
    for layer in base.layers:
        layer.trainable = (layer.name.startswith(prefixes)
                           and not isinstance(layer, layers.BatchNormalization))
    compile_model(model, lr)             # must recompile after changing trainable flags


def trainable_layer_names(model, base=None):
    names = [l.name for l in model.layers if l.trainable and l.name != getattr(base, "name", None)]
    if base is not None:
        names += [f"{base.name}/{l.name}" for l in base.layers if l.trainable]
    return names