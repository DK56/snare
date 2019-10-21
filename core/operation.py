import numpy as np
import math
import copy
from .model_wrapper import WeightsProvider
from .group import Group


class Operation(WeightsProvider):

    def __init__(self, weights_provider):
        self.weights_provider = weights_provider

    def get(self):
        return self.weights_provider.get()

    def update_config(self, config):
        return config


class NeuronPruner(Operation):

    def __init__(self, to_remove, weights_provider):
        self.to_remove = to_remove
        super().__init__(weights_provider)

    def get(self):
        weights = self.weights_provider.get()
        w = weights[0]
        b = weights[1]
        # print(self.to_remove.shape)
        # print(w.shape)
        # print(b.shape)

        w = np.delete(w, self.to_remove, w.ndim - 1)
        b = np.delete(b, self.to_remove)
        # print(w.shape)
        # print(b.shape)
        return [w, b]

    def update_config(self, config):
        updated = copy.deepcopy(config)
        if 'units' in updated:
            updated['units'] = updated['units'] - \
                len(self.to_remove)
        if 'filters' in config:
            updated['filters'] = updated['filters'] - \
                len(self.to_remove)
        return updated


class InputPruner(Operation):

    def __init__(self, to_remove, units, weights_provider):
        self.to_remove = to_remove
        self.units = units
        super().__init__(weights_provider)

    def get(self):
        # TODO could contain a major bug, but seems to work
        weights = self.weights_provider.get()
        w = weights[0]
        reshape = w.shape[w.ndim - 2] != self.units
        if reshape:
            w = w.reshape(int(w.shape[w.ndim-2] / self.units),
                          self.units, w.shape[w.ndim - 1])
        w = np.delete(w, self.to_remove, w.ndim - 2)
        if reshape:
            w = w.reshape(-1, w.shape[-1])
        return [w, weights[1]]


def prune_low_magnitude_neurons(group, percentages):
    base = group.base_wrapper
    main_layer = group.main_layer

    instances = []
    main_index = base.layers.index(main_layer)
    weights = main_layer.weights.get()

    w = weights[0]
    # b = weights[1]
    sums = w
    while(sums.ndim > 1):
        sums = sums.sum(axis=sums.ndim - 2)
    indices = np.argsort(sums)
    for p in percentages:
        to_remove = indices[0: math.ceil(p * indices.size)]
        print("Prune", int(p * 100), "%:",
              len(indices[0: math.ceil(p * indices.size)]), "neurons")
        instance = base.copy()

        main_layer = instance.layers[main_index]
        main_layer.apply_operation(
            NeuronPruner(to_remove, main_layer.weights))

        next_index = main_index + 1
        while not instance.layers[next_index].is_important():
            next_index += 1

        next_layer = instance.layers[next_index]
        next_layer.apply_operation(InputPruner(
            to_remove, indices.size, next_layer.weights))

        instances.append(instance)

    group.instances = instances
