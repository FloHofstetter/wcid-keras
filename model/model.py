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
    f = 3
    layers = [
        # Down1
        BatchNormalization(input_shape=input_shape),
        Conv2D(8, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        Conv2D(16, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        # Down2
        Conv2D(16, (int(f*5), int(f*5)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        Conv2D(32, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        Conv2D(32, (int(f*5), int(f*5)), padding="valid", strides=(1, 1), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        # Down3
        Conv2D(64, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        Conv2D(64, (int(f*5), int(f*5)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        MaxPooling2D(pool_size=(2, 2)),
        # Down4
        Conv2D(64, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        Conv2D(64, (int(f*5), int(f*5)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        MaxPooling2D(pool_size=(2, 2)),
        # Up1
        UpSampling2D(size=(2, 2)),
        Conv2DTranspose(64, (int(f*5), int(f*5)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        Conv2DTranspose(64, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        # Up2
        UpSampling2D(size=(2, 2)),
        Conv2DTranspose(64, (int(f*5), int(f*5)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        Conv2DTranspose(64, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        # Up3
        UpSampling2D(size=(2, 2)),
        Conv2DTranspose(32, (int(f*5), int(f*5)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        Conv2DTranspose(32, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        Conv2DTranspose(16, (int(f*5), int(f*5)), padding="valid", strides=(1, 1), activation="relu"),
        Dropout(0.2),
        # Up4
        UpSampling2D(size=(2, 2)),
        Conv2DTranspose(16, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="relu"),
        Conv2DTranspose(
            7, (int(f*3), int(f*3)), padding="valid", strides=(1, 1), activation="softmax"
        ),
    ]

    model = conv_model = keras.models.Sequential(layers)

    return model


def main():
    """

    :return:
    """
    model = get_model()
    model.summary()


if __name__ == "__main__":
    main()
