import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["PLAIDML_DEVICE_IDS"] = "llvm_cpu.0"

import keras
import json
import pathlib
import argparse


def convert(path):
    model_path = pathlib.Path(path, "tf_model.json")
    weights_path = pathlib.Path(path, "tf_model_weights.h5")

    json_paths = pathlib.Path.glob(path, "*.json")

    if not model_path.is_file:
        if len(json_paths) == 1:
            print(f"could not find tf_model.json, will try {json_paths}")
            model_path = json_paths

        if len(json_paths) > 1:
            msg = f"Could not find tf_model.json"
            raise ValueError(msg)

    if not weights_path.is_file:
        msg = "Could not find weigths"
        raise ValueError(msg)

    convertedJsonModel_path = pathlib.Path(path, "tf_to_plaidml_model.json")
    convertedModel_path = pathlib.Path(path, "tf_to_plaidml_model.h5")

    # python doesnt know what null is
    null = None

    # specific for the BoschNN
    replacement = {
        "class_name": "VarianceScaling",
        "config": {
            "scale": 1.0,
            "mode": "fan_avg",
            "distribution": "uniform",
            "seed": null,
        },
    }

    with open(model_path, "r") as json_data:
        data = json.load(json_data)

    for layers in data["config"]["layers"]:
        if "ragged" in layers["config"]:
            del layers["config"]["ragged"]

        if "groups" in layers["config"]:
            del layers["config"]["groups"]

        if "axis" in layers["config"]:
            if len(layers["config"]["axis"]) == 1:
                layers["config"]["axis"] = layers["config"]["axis"][0]

        if "kernel_initializer" in layers["config"]:
            layers["config"]["kernel_initializer"] = replacement

    with open(convertedJsonModel_path, "w") as out_file:
        json.dump(data, out_file)

    with open(convertedJsonModel_path, "r") as json_file:
        loaded_model_json = json_file.read()
        model = None
        try:
            model = keras.models.model_from_json(loaded_model_json)
        except:
            raise (f"{model_path} is not an acceptable model")

        model.load_weights(str(weights_path))
        model.summary()
        model.save(str(convertedModel_path))

    print("Converted Model saved to: {}".format(path))


def parse_args(parser: argparse.ArgumentParser):
    """
    Path to Tensoflor model.json and weights.h5

    :param parser: Argument parser Object.
    :return: CLI Arguments object.
    """
    parser.add_argument(
        "path",
        type=pathlib.Path,
        help="Path to Tensorflow model.json and weights.h5",
    )

    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    path = args.path

    convert(path)


if __name__ == "__main__":
    main()
