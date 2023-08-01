import tensorflow as tf

from model import model
import config as CFG

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load the tf.data.Dataset
train_ds = tf.data.Dataset.load(path="../data/Training")
train_ds = train_ds.batch(8).cache().prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.load(path="../data/Validation")
val_ds = val_ds.batch(8).cache().prefetch(tf.data.AUTOTUNE)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="C:\\Users\\abhi0\\Desktop\\Image_Search_Engine\\models\\resnet50v2_model-{epoch:02d}-{val_cosine_similarity:.2f}.hdf5",
    monitor="val_cosine_similarity",
    mode="max",
    save_best_only=True,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.CosineSimilarity(axis=1),
    metrics=[tf.keras.metrics.CosineSimilarity(axis=1)],
)

model.fit(x=train_ds, validation_data=val_ds, epochs=30, callbacks=[checkpoint])
