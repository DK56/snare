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


WeightsDict = Dict[str, WeightsProvider]


class ModelStructure():

    def __init__(self, order: List[str],
                 layer_configs, layer_weights: WeightsDict):
        self.order = order
        self.layer_configs = layer_configs
        self.layer_weights = layer_weights

    @classmethod
    def from_model(cls, model, path, suffix=''):
        order = list(map(lambda layer: layer.name, model.layers))

        layer_configs = {}

        for layer in model.layers:
            layer_configs[layer.name] = {
                'class_name': layer.__class__.__name__,
                'config': layer.get_config()}

        assert os.path.exists(path)
        assert os.path.isdir(path)

        layer_weights = {}
        if suffix:
            weights_file = 'weights_' + suffix + '.npy'
        else:
            weights_file = 'weights.npy'

        for layer in model.layers:
            layer_dir = os.path.join(path, layer.name)
            if not os.path.exists(layer_dir):
                os.mkdir(layer_dir)

            w = FileWeights(os.path.join(layer_dir, weights_file))
            w.save(layer.get_weights())
            layer_weights[layer.name] = w

        return cls(order, layer_configs, layer_weights)

    @staticmethod
    def _save_configs(configs, path, suffix=None):
        assert os.path.exists(path)
        assert os.path.isdir(path)

        if suffix:
            config_file = 'config_' + suffix + '.json'
        else:
            config_file = 'config.json'

        for layer_name, config in configs:
            layer_dir = os.path.join(path, layer_name)
            if not os.path.exists(layer_dir):
                os.mkdir(layer_dir)

            config_path = os.path.join(layer_dir, config_file)
            assert not os.path.exists(config_path)

            config_json = json.dumps(config)

            with open(config_path, 'w+') as f:
                f.write(config_json)

    @staticmethod
    def _save_weights(weights, path, suffix=None):
        assert os.path.exists(path)
        assert os.path.isdir(path)

        if suffix:
            weights_file = 'weights_' + suffix + '.npy'
        else:
            weights_file = 'weights.npy'

        for layer_name, layer_weights in weights:
            layer_dir = os.path.join(path, layer_name)
            if not os.path.exists(layer_dir):
                os.mkdir(layer_dir)

            w = FileWeights(os.path.join(layer_dir, weights_file))
            w.save(layer_weights.get())

    def to_model(self):
        model = Sequential()
        for layername in self.order:
            weights = self.layer_weights[layername].get()
            layer = layers.deserialize(self.layer_configs[layername])

            model.add(layer)
            layer.set_weights(weights)
        return model

    def save_configs(self, path, suffix=None):
        self._save_configs(self.layer_configs, path, suffix)

    def save_weights(self, path, suffix=None):
        self._save_weights(self.layer_weights, path, suffix)
