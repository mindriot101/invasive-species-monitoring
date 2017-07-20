#!/usr/bin/env python
# coding: utf-8


from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import logging
import currentmodel

N_EPOCHS = 50

# Read the labels from the supplied labels file
train_labels = pd.read_csv('train_labels.csv')
train_labels.head()
y = train_labels.invasive.values  # classification labels (0 or 1)

# Load the training data set. Has the shape:
#  (2295, 224, 224, 3)
# 2295 images of 224x224 pixels with 3 colours
X = np.load('train.npy')

assert X.shape[0] == y.shape[0]

# Split the datasets into 75% training and 25% validation
X_train, X_test, y_train, y_test = train_test_split(X, y)

# See currentmodel.py: https://github.com/mindriot101/invasive-species-monitoring/blob/master/currentmodel.py
model = currentmodel.build_model()
# Print a nice summary of the layers involved
model.summary()

# Fancy image generator which takes the source images, and randomly adjusts the
# image, to add extra training information. This rotates the images by up to 30
# degrees, and shifts them 10% left/right and up/down, and sometimes flips the
# images horizontally. This has the effect of increasing the size of the
# training dataset, without requiring extra data collection
#
# The `flow` method is an infinite stream of image data with shape
#  (32, 224, 224, 3)
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
train_datagen.fit(X_train)

# Fit the model, by iterating on the input dataset. The callbacks parameter
# means the model and its weights are saved to disk after every epoch
model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=X_train.shape[0] // 32,
    epochs=N_EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model',
                               monitor='val_acc', save_best_only=True)],
)
