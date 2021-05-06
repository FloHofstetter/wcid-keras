import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow
from tensorflow import keras
import pathlib

# get paths
# base_path = "/data/hofstetter_data/bosch_testset/prediction/mark_012/train/"
base_path = "/data/hofstetter_data/bosch_testset/prediction/mark_002/"
split = base_path.split("/")

for splinter in split:
    if "mark" in splinter:
        mark_number = splinter

model_path = pathlib.PurePath(base_path, "end_model.h5")
save_dir = pathlib.PurePath("/data/chumak_data/tf_to_plaidml/", mark_number)

# create save dir if nonexistent
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

# load model as json
model = keras.models.load_model(str(model_path))
model_json = model.to_json()

jsonModel_save_path = pathlib.PurePath(save_dir ,"tf_model.json")
weights_save_path = pathlib.PurePath(save_dir ,"tf_model_weights.h5")

with open(jsonModel_save_path, "w") as json_file:
    json_file.write(model_json)

model.save_weights(weights_save_path)
print("Tensorflow model saved to: {}".format(save_dir))
