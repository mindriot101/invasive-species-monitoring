
# coding: utf-8

# In[ ]:


from keras.layers import *
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import h5py
import logging
import currentmodel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('imagenet')

N_EPOCHS = 50

# In[ ]:


logger.info('Reading labels')
train_labels = pd.read_csv('train_labels.csv')
train_labels.head()


# In[ ]:


y = train_labels.invasive.values
# y = to_categorical(y)


# In[ ]:


logger.info('Loading features')
X = np.load('train.npy')


# In[ ]:


assert X.shape[0] == y.shape[0]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


# In[ ]:

logger.info('Setting up model')
model = currentmodel.build_model()

# In[ ]:


model.summary()


# In[ ]:


logger.info('Building image data generator')
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
train_datagen.fit(X_train)


# In[ ]:


logger.info('Fitting model')
model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=X_train.shape[0] // 32,
    epochs=N_EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc',
        save_best_only=True)],
)
