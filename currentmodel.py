from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import *

def build_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


    # In[ ]:


    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(1, activation='sigmoid'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1E-4, momentum=0.9),
            metrics=['accuracy'])

    return model
