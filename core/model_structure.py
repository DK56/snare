import os
import json
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from typing import List, Dict

PathDict = Dict[str, os.PathLike]


class ModelStructure():

    def __init__(self, order: List[str], layers: PathDict, weights: PathDict):
        self.order = order
        self.layers = layers
        self.weights = weights

    def to_model(self):
        model = Sequential()
        for layername in self.order:

            layer_config_path = self.layers[layername]
            assert os.path.exists(layer_config_path)
            assert os.path.isfile(layer_config_path)
            with open(layer_config_path) as file:
                layer_config = json.load(file)
            assert layer_config

            weights_path = self.weights[layername]
            assert os.path.exists(weights_path)
            assert os.path.isfile(weights_path)
            with open(weights_path, 'rb+') as file:
                data = np.load(file)
                weights = [value for (key, value) in sorted(data.items())]

            layer = layers.deserialize(layer_config)
            model.add(layer)
            layer.set_weights(weights)
            print(layer_config)
        return model
