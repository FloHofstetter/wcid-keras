from dataset import RailDataset
from model import get_model
import tensorflow as tf
import pathlib


def train(trn_img, val_img, trn_msk, val_msk, epochs=1, save_pth="checkpoints/"):
    """
    Tran the model an save checkpoints along the training.

    :return:
    """
    # Data generator
    trn_gen = RailDataset(trn_img, trn_msk, (320, 160), "png", "png", batch_size=1)
    val_gen = RailDataset(val_img, val_msk, (320, 160), "png", "png", batch_size=1)

    model = get_model()

    # Compile model
    loss = tf.keras.losses.binary_crossentropy
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    # Checkpoint callback
    pathlib.Path(save_pth).mkdir(parents=True, exist_ok=True)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_pth,
                                                     save_weights_only=True,
                                                     verbose=1)

    # Train model
    model.fit(trn_gen, validation_data=val_gen, epochs=epochs, callbacks=[cp_callback])

    # Save Checkpoints
    model.save("model.h5")


def main():
    trn_img = ""
    val_img = ""
    trn_msk = ""
    val_msk = ""
    train(trn_img, val_img, trn_msk, val_msk, epochs=10)


if __name__ == '__main__':
    main()
