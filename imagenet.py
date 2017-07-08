
# coding: utf-8

# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import *
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from scipy.misc import imread, imresize
from keras.preprocessing import image
import h5py


# In[ ]:


train_labels = pd.read_csv('train_labels.csv')
train_labels.head()


# In[ ]:


y = train_labels.invasive.values
# y = to_categorical(y)


# In[ ]:


X = np.load('train.npy')


# In[ ]:


assert X.shape[0] == y.shape[0]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


# In[ ]:


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.summary()


# In[ ]:


add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

for layer in base_model.layers:
    layer.trainable = False

model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
train_datagen.fit(X_train)


# In[ ]:


model.fit_generator(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=X_train.shape[0] // 32,
    epochs=1,
    validation_data=(X_test, y_test)
)

model.save('trained_model.h5')
