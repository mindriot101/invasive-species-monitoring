from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import *

def build_model():
    '''
    As a shortcut, take the existing VGG16 model[1] from the Oxford group, which
    has been pre-trained on the imagenet classification dataset. By setting
    `include_only=False` we only include the pre-trained convolutional layers,
    and not the neural network layers.

    We then add our own convolutional layers on top, and optimise with SGD.

    [1]: https://arxiv.org/abs/1409.1556
    '''
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add our own neural network layers
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(1, activation='sigmoid'))

    # Combine the VGG16 and our own layers
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1E-4, momentum=0.9),
            metrics=['accuracy'])

    return model
