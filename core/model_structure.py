import os
import json
import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from typing import List, Dict


class WeightsProvider(ABC):
    @abstractmethod
    def get(self):
        pass


class FileWeights(WeightsProvider):
    def __init__(self, path):
        self.path = path

    def save(self, weights):
        assert not os.path.exists(self.path)

        with open(self.path, 'wb+') as f:
            np.savez(f, *weights)

    def get(self):
        assert os.path.exists(self.path)
        assert os.path.isfile(self.path)
        with open(self.path, 'rb+') as file:
            data = np.load(file)
            weights = [value for (key, value) in sorted(data.items())]
        return weights


PathDict = Dict[str, os.PathLike]
WeightsDict = Dict[str, WeightsProvider]


class ModelStructure():

    def __init__(self, order: List[str],
                 layers: PathDict, weights: WeightsDict):
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

            weights = self.weights[layername].get()
            layer = layers.deserialize(layer_config)

            model.add(layer)
            layer.set_weights(weights)
            print(layer_config)
        return model
