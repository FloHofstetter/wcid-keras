from tensorflow import keras
from tensorflow.keras.layers import Dropout, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np


def get_model(input_shape=(320, 160)):
    """
    Get the CNN network model

    :param input_shape: Shape of the input image.
    :return: Network model.
    """
    # Change width and height for keras and append channels
    input_shape = list(input_shape)
    input_shape.append(3)
    input_shape[0], input_shape[1] = input_shape[1], input_shape[0]

    # Create model
    model = keras.models.Sequential()

    # Down 1
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Down 2
    model.add(Conv2D(16, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Down 3
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Down 4
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Up 1
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(64, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))

    # Up 2
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(64, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))

    # Up 3
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(32, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2DTranspose(16, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Dropout(0.2))

    # Up 4
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu'))
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='sigmoid'))

    return model
