import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from .model_wrapper import ModelWrapper


class Group():
    IMPORTANT_LAYERS = ['Conv2D', 'Dense']

    def __init__(self, main_layer, wrapper):
        self.main_layer = main_layer
        self.base_wrapper = wrapper

    @classmethod
    def create_groups(cls, model_wrapper, min_group_size, offset=0):
        assert min_group_size > 0
        # TODO Needs to be an deepcopy
        wrapper = model_wrapper

        groups = []

        processable_layers = []
        for i, layer_name in enumerate(wrapper.order):
            layer_config = wrapper.layer_configs[layer_name]

            if layer_config['class_name'] in ['Conv2D', 'Dense']:
                processable_layers.append(i)

        start = 0
        end = 0
        current_group_size = 0

        max_end = len(wrapper.order)
        print(processable_layers)

        for i, index in enumerate(processable_layers):
            if index < offset:
                continue

            current_group_size += 1
            if current_group_size == 1:
                main_layer = wrapper.order[index]

            if current_group_size == min_group_size:
                if i + min_group_size >= len(processable_layers):
                    end = max_end
                else:
                    end = processable_layers[i + 1]

                group = cls(main_layer, ModelWrapper.from_model_wrapper(
                    wrapper, start, end))
                groups.append(group)

                start = end
                current_group_size = 0
                if end == max_end:
                    break

        return groups
