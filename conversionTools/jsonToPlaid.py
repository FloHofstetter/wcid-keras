import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["PLAIDML_DEVICE_IDS"] = "opencl_nvidia_geforce_rtx_2080_ti.1"
import keras
import json
import pathlib

base_path = "/data/chumak_data/tf_to_plaidml/mark_002"

model_path = pathlib.Path(base_path, "tf_model.json")
weights_path = pathlib.Path(base_path, "tf_model_weights.h5")

convertedJsonModel_path = pathlib.Path(base_path, "tf_to_plaidml_model.json")
convertedModel_path = pathlib.Path(base_path, "tf_to_plaidml_model.h5")

null = None

replacement = {
                "class_name": "VarianceScaling",
                "config": {
                        "scale": 1.0,
                        "mode": "fan_avg",
                        "distribution": "uniform",
                        "seed": null
                    }
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
    model = keras.models.model_from_json(loaded_model_json)
    model.load_weights(str(weights_path))
    model.summary()
    model.save(str(convertedModel_path))

print("Converted Model saved to: {}".format(base_path))