import tensorflow as tf

# Get the pretrained model
# pretrained_head = tf.keras.applications.VGG16(
#     include_top=True, input_shape=(224, 224, 3)
# )
# last_layer = pretrained_head.get_layer("fc2")

# x = last_layer.output
# x = tf.keras.layers.Dense(2000, name="intermediate_layer")(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(rate=0.5)(x)
# x = tf.keras.layers.Dense(300, name="embedding_layer")(x)
# embed_output = tf.keras.layers.BatchNormalization()(x)

# model = tf.keras.Model(
#     inputs=pretrained_head.input, outputs=embed_output, name="Image_Embedding_model"
# )
# print(model.summary())


pretrained_head = tf.keras.applications.ResNet50V2(
    include_top=True, input_shape=(224, 224, 3)
)
last_layer = pretrained_head.get_layer(name="avg_pool")
x = last_layer.output
x = tf.keras.layers.Dense(2000, name="intermediate_layer")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(rate=0.5)(x)
x = tf.keras.layers.Dense(300, name="embedding_layer")(x)
embed_output = tf.keras.layers.BatchNormalization()(x)

model = tf.keras.Model(
    inputs=pretrained_head.input, outputs=embed_output, name="Image_Embedding_model"
)
