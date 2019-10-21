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


class LayerWrapper():
    IMPORTANT_LAYERS = ['Conv1D', 'Conv2D', 'Dense']

    def __init__(self, name, classname, config,
                 input_shape, output_shape, weights: WeightsDict):
        self.name = name
        self.classname = classname
        self.config = config
        self.weights = weights
        self.input_shape = input_shape
        self.output_shape = output_shape

    @classmethod
    def from_layer(cls, layer, path, suffix=''):

        assert os.path.exists(path)
        assert os.path.isdir(path)

        name = layer.name
        classname = layer.__class__.__name__

        if suffix:
            weights_file = 'weights_' + suffix + '.npy'
        else:
            weights_file = 'weights.npy'

        layer_dir = os.path.join(path, name)
        if not os.path.exists(layer_dir):
            os.mkdir(layer_dir)

        w = FileWeights(os.path.join(layer_dir, weights_file))
        w.save(layer.get_weights())

        return cls(name, classname, layer.get_config(),
                   layer.input_shape[1:], layer.output_shape[1:], w)

    def is_important(self):
        return self.classname in LayerWrapper.IMPORTANT_LAYERS

    def to_layer(self):
        return layers.deserialize({
            'class_name': self.classname,
            'config': self.config})

    def apply_operation(self, op):
        self.weights = op
        self.config = op.update_config(self.config)
        # TODO update input/output_shape

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.name, self.classname, self.config.copy(),
                          self.input_shape, self.output_shape, self.weights)

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not(self == other)


class ModelWrapper():

    def __init__(self, layers):
        self.layers = layers

    @classmethod
    def from_model(cls, model, path, suffix=''):
        # assert Sequential

        assert os.path.exists(path)
        assert os.path.isdir(path)

        l_wrappers = [LayerWrapper.from_layer(
            layer, path, suffix) for layer in model.layers]

        return cls(l_wrappers)

    @classmethod
    def from_model_wrapper(cls, wrapper, start, end):

        layers = []

        for i in range(start, end):
            l_wrapper = wrapper.layers[i]
            layers.append(l_wrapper.copy())

        return cls(layers)

    # @staticmethod
    # def _save_configs(configs, path, suffix=None):
    #     assert os.path.exists(path)
    #     assert os.path.isdir(path)

    #     if suffix:
    #         config_file = 'config_' + suffix + '.json'
    #     else:
    #         config_file = 'config.json'

    #     for layer_name, config in configs:
    #         layer_dir = os.path.join(path, layer_name)
    #         if not os.path.exists(layer_dir):
    #             os.mkdir(layer_dir)

    #         config_path = os.path.join(layer_dir, config_file)
    #         assert not os.path.exists(config_path)

    #         config_json = json.dumps(config)

    #         with open(config_path, 'w+') as f:
    #             f.write(config_json)

    # @staticmethod
    # def _save_weights(weights, path, suffix=None):
    #     assert os.path.exists(path)
    #     assert os.path.isdir(path)

    #     if suffix:
    #         weights_file = 'weights_' + suffix + '.npy'
    #     else:
    #         weights_file = 'weights.npy'

    #     for layer_name, layer_weights in weights:
    #         layer_dir = os.path.join(path, layer_name)
    #         if not os.path.exists(layer_dir):
    #             os.mkdir(layer_dir)

    #         w = FileWeights(os.path.join(layer_dir, weights_file))
    #         w.save(layer_weights.get())

    # def save_configs(self, path, suffix=None):
    #     self._save_configs(self.layer_configs, path, suffix)

    # def save_weights(self, path, suffix=None):
    #     self._save_weights(self.layer_weights, path, suffix)

    def to_model(self):
        model = Sequential()

        if 'batch_input_shape' not in self.layers[0].config:
            model.add(layers.Input(shape=self.layers[0].input_shape))

        for l_wrapper in self.layers:
            layer = l_wrapper.to_layer()
            model.add(layer)
            weights = l_wrapper.weights.get()
            layer.set_weights(weights)

        return model

    def update(self, to_update):
        start = self.layers.index(to_update.layers[0])
        for i in range(len(to_update.layers)):
            self.layers[start + i] = to_update.layers[i].copy()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        layers = [layer.copy() for layer in self.layers]
        return type(self)(layers)
