from dataset import RailDataset
from model import get_model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os


def train(
    trn_img,
    val_img,
    trn_msk,
    val_msk,
    train_res=(320, 160),
    epochs=1,
):
    """
    Tran the model, save model and weights.

    :param trn_img: Path to training image folder.
    :param val_img: Path to validation image folder.
    :param trn_msk: Path to training masks.
    :param val_msk: Path to validation masks.
    :param train_res: Resolution to train the network. (width, height)
    :param epochs: Training epochs to iterate over dataset.
    :return: None.
    """
    # Data generator
    trn_gen = RailDataset(trn_img, trn_msk, train_res, "png", "png", batch_size=1)
    val_gen = RailDataset(val_img, val_msk, train_res, "png", "png", batch_size=1)

    model = get_model(input_shape=train_res)

    # Compile model
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,  # 0.00001
        decay_steps=10000,
        decay_rate=0.9,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])

    # Checkpoint callback
    cp_pth = "checkpoints/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_pth, save_weights_only=True, verbose=1
    )

    # Train model
    history = model.fit(
        trn_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[cp_callback],
        workers=48,
        use_multiprocessing=False,
    )

    # Save Checkpoints
    model.save("model.h5")

    # Show history
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig("history.svg")


def main():
    """
    Entry point for training.
    """
    # Select GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    trn_img = "/data/hofstetter_data/nlb/img/trn/"
    val_img = "/data/hofstetter_data/nlb/img/val/"
    trn_msk = "/data/hofstetter_data/nlb/msk/trn_t/"
    val_msk = "/data/hofstetter_data/nlb/msk/val_t/"
    train_res = (320, 160)  # Width height
    train(trn_img, val_img, trn_msk, val_msk, epochs=50, train_res=train_res)


if __name__ == "__main__":
    main()
