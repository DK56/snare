import os
import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from typing import Dict


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

        name = layer.__class__.__name__
        if name == 'Dense':
            return layer.units * (2 + layer.input_shape[1])
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

            # first dim
            num_instances_per_filter = (
                (rows - layer.kernel_size[0] + 1) / layer.strides[0]) + 1
            # second dim
            num_instances_per_filter *= (
                (cols - layer.kernel_size[1] + 1) / layer.strides[1]) + 1

            flops_per_filter = num_instances_per_filter * ops
            return layer.filters * flops_per_filter

        IRRELEVANT_LAYERS = ['Flatten', 'InputLayer', 'AveragePooling1D',
                             'AveragePooling2D', 'AveragePooling3D',
                             'MaxPooling1D', 'MaxPooling2D', 'MaxPooling3D',
                             'BatchNormalization', 'Dropout', 'Activation']
        if name in IRRELEVANT_LAYERS:
            # Skip all layers, which do not influence flops or only a small
            # contribution compared to all prunable layers
            return 0

        print("Unsupported layer", name)
        return 0

    def __init__(self, name, classname, config,
                 neurons, params, flops, input_shape,
                 output_shape, weights: WeightsDict):
        self.name = name
        self.classname = classname
        self.config = config
        self.weights = weights
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.neurons = neurons
        self.params = params
        self.flops = flops

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
        weights = layer.get_weights()
        w.save(weights)
        config = layer.get_config()
        if "units" in config:
            neurons = layer.units
        elif "filters" in config:
            neurons = layer.filters
        else:
            neurons = 0
        params = 0
        for p in weights:
            params += p.size

        return cls(name, classname, config,
                   neurons, params, LayerWrapper.get_flops(layer),
                   layer.input_shape[1:], layer.output_shape[1:], w)

    def is_important(self):
        return self.classname in LayerWrapper.IMPORTANT_LAYERS

    def is_batch_norm(self):
        return self.classname in LayerWrapper.BATCH_NORM_LAYERS

    def to_layer(self):
        if self.classname == "CustomConnected":
            return layers.deserialize({
                'class_name': "Dense",
                'config': self.config})
        if self.classname == "CustomConv":
            return layers.deserialize({
                'class_name': "Conv1D",
                'config': self.config})
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
                          self.neurons, self.params, self.flops,
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

    def __init__(self, layers, compile_args):
        self.layers = layers
        self.compile_args = compile_args

    @classmethod
    def from_model(cls, model, compile_args, path, suffix=''):
        # assert Sequential

        assert os.path.exists(path)
        assert os.path.isdir(path)

        l_wrappers = [LayerWrapper.from_layer(
            layer, path, suffix) for layer in model.layers]

        return cls(l_wrappers, compile_args)

    @classmethod
    def from_model_wrapper(cls, wrapper, start, end):

        layers = []

        for i in range(start, end):
            l_wrapper = wrapper.layers[i]
            layers.append(l_wrapper.copy())

        if start == 0 and len(wrapper.layers) == end:
            return cls(layers, wrapper.compile_args)
        else:
            compile_args = wrapper.compile_args.copy()
            compile_args['loss'] = 'mse'
            return cls(layers, compile_args)

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

        for i, l_wrapper in enumerate(self.layers):
            layer = l_wrapper.to_layer()
            model.add(layer)
            weights = l_wrapper.weights.get()
            layer.set_weights(weights)

        model.compile(**self.compile_args)
        return model

    def to_splitted_model(self, split):
        # TODO check compile_args
        m1 = Sequential()

        if 'batch_input_shape' not in self.layers[0].config:
            m1.add(layers.Input(shape=self.layers[0].input_shape))

        for i, l_wrapper in enumerate(self.layers):
            if i == split:
                break
            layer = l_wrapper.to_layer()
            m1.add(layer)
            weights = l_wrapper.weights.get()
            layer.set_weights(weights)

        m2 = Sequential()
        m2.add(layers.Input(m1.output_shape[1:]))

        for i, l_wrapper in enumerate(self.layers):
            if i < split:
                continue
            layer = l_wrapper.to_layer()
            m2.add(layer)
            weights = l_wrapper.weights.get()
            layer.set_weights(weights)

        return m1, m2

    def to_trainable_model(self):
        model = Sequential()

        if 'batch_input_shape' not in self.layers[0].config:
            model.add(layers.Input(shape=self.layers[0].input_shape))

        for l_wrapper in self.layers:
            layer = l_wrapper.to_layer()
            # if layer.name == "dense":
            #     weights = l_wrapper.weights.get()
            #     w = weights[0]
            #     mask = ((w > 0.001) | (w < -0.001)).astype(int)
            #     print(mask)

            #     layer = CustomConnected(layer.units, mask, name="dense")

            if layer.name == "conv1d":
                if 'batch_input_shape' in self.layers[0].config:
                    model.add(layers.Input(shape=self.layers[0].input_shape))
                weights = l_wrapper.weights.get()
                w = weights[0]
                mask = ((w > 0.001) | (w < -0.001)).astype(int)

                layer = CustomConv(
                    layer.filters, layer.kernel_size, mask,
                    padding=layer.padding, name=layer.name)
            model.add(layer)
            weights = l_wrapper.weights.get()
            layer.set_weights(weights)
        model.compile(**self.compile_args)
        return model

    def update(self, to_update):
        # TODO
        # Update compile_args only if both models have equal tamount of layers
        #   self.compile_args = to_update.compile_args

        start = self.layers.index(to_update.layers[0])
        for i in range(len(to_update.layers)):
            self.layers[start + i] = to_update.layers[i].copy()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        layers = [layer.copy() for layer in self.layers]
        compile_args = self.compile_args.copy()
        return type(self)(layers, compile_args)
