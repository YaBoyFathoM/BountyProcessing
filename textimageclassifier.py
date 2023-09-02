import os
import tensorflow as tf
import sys
from PIL import Image

layers = tf.keras.layers

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


dataset = tf.data.Dataset.load("binary_image_dataset")

# Preprocess the dataset
preprocessed_dataset = dataset.map(lambda x, y: (tf.expand_dims(x, axis=0), tf.expand_dims(y, axis=0)))


model.fit(preprocessed_dataset, epochs=5)
#model.save("chat_cleaner.h5")

