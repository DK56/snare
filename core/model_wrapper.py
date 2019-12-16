import os
import json
import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from typing import List, Dict
from tensorflow.python.keras import backend as K


class CustomConv(layers.Conv1D):

    def __init__(self, filters, kernel_size, connections, **kwargs):

        # this is matrix A
        self.connections = connections

        # initalize the original Dense with all the usual arguments
        print("SHAPE:", self.connections.shape)
        super(CustomConv, self).__init__(filters, kernel_size, **kwargs)

    def call(self, inputs):
        masked_kernel = self.kernel * self.connections
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                masked_kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                masked_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                masked_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class CustomConnected(layers.Dense):

    def __init__(self, units, connections, **kwargs):

        # this is matrix A
        self.connections = connections

        # initalize the original Dense with all the usual arguments
        super(CustomConnected, self).__init__(units, **kwargs)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


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
    IMPORTANT_LAYERS = ['Conv1D', 'Conv2D',
                        'Dense', 'CustomConnected', 'CustomConv']
    BATCH_NORM_LAYERS = ['BatchNormalization']

    def get_flops(layer):
        if layer.__class__.__name__ == 'Dense':
            return layer.units * (2 + layer.input_shape[1])
        if layer.__class__.__name__ == 'Flatten':
            return 0
        if layer.__class__.__name__ == 'InputLayer':
            return 0
        if layer.__class__.__name__ == 'AveragePooling2D':
            return 0
        if layer.__class__.__name__ == 'MaxPooling1D':
            return 0
        if layer.__class__.__name__ == 'MaxPooling2D':
            return 0
        if layer.__class__.__name__ == 'Conv1D':
            input_shape = layer.input_shape
            if layer.data_format == "channels_last":
                channels = input_shape[2]
                rows = input_shape[1]
            else:
                channels = input_shape[1]
                rows = input_shape[2]

            ops = (channels + rows) * 2 - 1

            num_instances_per_filter = (
                (rows - layer.kernel_size[0] + 1) / layer.strides[0]) + 1

            flops_per_filter = num_instances_per_filter * ops
            return layer.filters * flops_per_filter
        if layer.__class__.__name__ == 'BatchNormalization':
            return 0
        if layer.__class__.__name__ == 'Dropout':
            return 0
        if layer.__class__.__name__ == 'Activation':
            return 0
        if layer.__class__.__name__ == 'Conv2D':
            input_shape = layer.input_shape
            if layer.data_format == "channels_last":
                channels = input_shape[3]
                rows = input_shape[1]
                cols = input_shape[2]
            else:
                channels = input_shape[1]
                rows = input_shape[2]
                cols = input_shape[3]

            ops = (channels + rows + cols) * 2 - 1

            num_instances_per_filter = (
                (rows - layer.kernel_size[0] + 1) / layer.strides[0]) + 1  # for rows
            num_instances_per_filter *= (
                (cols - layer.kernel_size[1] + 1) / layer.strides[1]) + 1

            flops_per_filter = num_instances_per_filter * ops
            return layer.filters * flops_per_filter

        print("Unsupported layer", layer.__class__.__name__)
        return 0

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
