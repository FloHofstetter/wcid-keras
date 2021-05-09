import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
import pathlib
import argparse


def toJSON(model_path: pathlib.Path, save_dir: pathlib.Path):

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    if model_path.is_dir:
        model_path = pathlib.Path(model_path, "end_model.h5")

    # load model as json
    model = keras.models.load_model(str(model_path))
    model_json = model.to_json()

    jsonModel_save_path = pathlib.Path(save_dir, "tf_model.json")
    weights_save_path = pathlib.Path(save_dir, "tf_model_weights.h5")

    with open(jsonModel_save_path, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(weights_save_path)
    print("Tensorflow model saved to: {}".format(save_dir))


def parse_args(parser: argparse.ArgumentParser):
    """
    Path to Tensoflor model.json and weights.h5

    :param parser: Argument parser Object.
    :return: CLI Arguments object.
    """
    parser.add_argument(
        "model_path",
        type=pathlib.Path,
        help="Path to Tensorflow’s end_model.h5",
    )
    parser.add_argument(
        "save_dir",
        type=pathlib.Path,
        help="Output path for model.json and weight.h5",
    )

    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    model_path = args.model_path
    save_dir = args.save_dir

    toJSON(model_path, save_dir)


if __name__ == "__main__":
    main()
