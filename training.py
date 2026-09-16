import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
h
model = MobileNetV2(weights = "imagenet", include_top=False)
model.trainable = False

#load photos and assigns them a marker based on class
train_ds = tf.keras.utils.image_dataset_from_directory(r"C:\Users\anshi\Desktop\garbage sorting\data\Raw Data\Training Dataset", image_size=(224, 224), batch_size = 32)

val_ds = tf.keras.utils.image_dataset_from_directory(r"C:\Users\anshi\Desktop\garbage sorting\data\Raw Data\Validation Dataset", image_size=(224, 224), batch_size = 32)

#processes the photos to adapt them to the model requirements
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))

val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

full_model = Sequential([
    model,
    GlobalAveragePooling2D(),
    Dense(4, activation='softmax')
])

full_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = full_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)
