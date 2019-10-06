import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from .model_wrapper import *


class Group():
    IMPORTANT_LAYERS = ['Conv2D', 'Dense']

    def __init__(self, main_layer, wrapper):
        self.main_layer = main_layer
        self.base_wrapper = wrapper

    @classmethod
    def create_groups(cls, model_structure, min_group_size, offset=0):
        assert min_group_size > 0
        # TODO Needs to be an deepcopy
        s = model_structure
        groups = []

        first = True

        important_layers = 0
        group_order = []
        group_layers = {}
        group_weights = {}

        for i, layer_name in enumerate(s.order):
            layer_config = s.layer_configs[layer_name]

            if i >= offset \
                    and layer_config['class_name'] in ['Conv2D', 'Dense']:

                # TODO could be an argument of create_groups
                if important_layers == 0:
                    group_main_layer = layer_name

                if important_layers == min_group_size or \
                        (important_layers + 1 == min_group_size
                         and i == len(s.order) - 1):
                            
                    group = cls(group_main_layer, group_order,
                                group_layers, group_weights, first)
                    if first:
                        group.has_input_shape = False
                        group.input_shape = \
                            s.layer_output_shapes[s.order[i - 1]]
                        first = False
                    groups.append(group)

                    important_layers = 0
                    group_order = []
                    group_layers = {}
                    group_weights = {}

                important_layers += 1

            group_order.append(layer_name)
            group_layers[layer_name] = layer_config
            group_weights[layer_name] = s.layer_weights[layer_name]

        # TODO bug with last layer and min_group_size = 1
        if len(group_order) != 0:
            groups[-1].order.extend(group_order)
            groups[-1].layer_configs.update(group_layers)
            groups[-1].layer_weights.update(group_weights)

        return groups

    def to_model(self):
        model = Sequential()
        if not self.has_input_shape:
            model.add(layers.InputLayer(self.input_shape))
        for layer_name in self.order:
            weights = self.layer_weights[layer_name].get()
            layer = layers.deserialize(self.layer_configs[layer_name])

            model.add(layer)
            layer.set_weights(weights)
        return model

    def to_full_model(self, model_structure):
        # TODO deepcopy instead of update
        model_structure.layer_weights.update(self.layer_weights)
        model_structure.layer_configs.update(self.layer_configs)
        return model_structure.to_model()
