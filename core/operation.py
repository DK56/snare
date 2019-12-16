import numpy as np
import math
import copy
from .model_wrapper import WeightsProvider
from .group import Group
from tensorflow.python.keras import backend as K


class Operation(WeightsProvider):

    def __init__(self, weights_provider):
        self.weights_provider = weights_provider

    def get(self):
        return self.weights_provider.get()

    def update_config(self, config):
        return config


class ConnectionPruner(Operation):

    def __init__(self, to_remove, weights_provider):
        self.to_remove = to_remove
        super().__init__(weights_provider)

    def get(self):
        weights = self.weights_provider.get()
        w = weights[0]
        shape = w.shape
        w = w.flatten()
        w[self.to_remove] = 0
        w = w.reshape(shape)
        return [w, weights[1]]

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
    randoms = []

    def __init__(self, to_remove, units, weights_provider):
        self.to_remove = to_remove
        self.units = units
        super().__init__(weights_provider)

    def get(self):
        # TODO could contain a major bug, but seems to work
        weights = self.weights_provider.get()

        # TODO filter BatchNormalization correctly
        if(len(weights) == 4):
            res = []
            for w in weights:
                res.append(np.delete(w, self.to_remove))
            return res

        w = weights[0]
        reshape = w.shape[w.ndim - 2] != self.units
        if reshape:
            w = w.reshape(int(w.shape[w.ndim-2] / self.units),
                          self.units, w.shape[w.ndim - 1])
        w = np.delete(w, self.to_remove, w.ndim - 2)
        if reshape:
            w = w.reshape(-1, w.shape[-1])
        return [w, weights[1]]


def _prune_connections(group, percentages, indices, weights):
    base = group.base_wrapper
    main_layer = group.main_layer
    main_index = base.layers.index(main_layer)
    instances = []
    w = weights[0]

    to_remove = 5
    param_amount = np.count_nonzero(w)

    print("SIZE:", w.size, "ZEROS:", w.size - np.count_nonzero(w))

    for p in percentages:
        to_remove = indices[0: w.size -
                            param_amount + to_remove]
        print(to_remove.shape)
        print(to_remove)
        # param_amount + math.ceil(p * param_amount)]
        print(w.size - param_amount + math.ceil(p * param_amount),
              " / ", param_amount, " / ", w.size)
        instance = base.copy()

        main_layer = instance.layers[main_index]
        main_layer.apply_operation(
            ConnectionPruner(to_remove, main_layer.weights))

        instances.append(instance)

    group.instances = instances


def prune_low_magnitude_connections(group, percentages):
    weights = group.main_layer.weights.get()

    w = weights[0]
    # b = weights[1]

    w_abs = np.abs(w)
    indices = np.argsort(w_abs.flatten())

    _prune_connections(group, percentages, indices, weights)


def prune_random_connections(group, percentages):
    weights = group.main_layer.weights.get()

    w = weights[0]
    # b = weights[1]

    if(InputPruner.randoms == []):
        InputPruner.randoms = np.arange(w.size)
        np.random.shuffle(InputPruner.randoms)

    _prune_connections(group, percentages, InputPruner.randoms, weights)


def _prune_neurons(group, percentages, indices):
    main_layer = group.main_layer
    base = group.base_wrapper
    instances = []

    main_index = base.layers.index(main_layer)

    for p in percentages:
        to_remove = indices[0: math.ceil(p * indices.size)]
        print("Prune", int(p * 100), "%:",
              len(indices[0: math.ceil(p * indices.size)]), "neurons")
        instance = base.copy()

        main_layer = instance.layers[main_index]
        main_layer.apply_operation(
            NeuronPruner(to_remove, main_layer.weights))

        next_index = main_index + 1
        next_layer = instance.layers[next_index]
        while not next_layer.is_important():
            if next_layer.is_batch_norm():
                next_layer.apply_operation(InputPruner(
                    to_remove, indices.size, next_layer.weights))

            next_index += 1
            next_layer = instance.layers[next_index]

        next_layer = instance.layers[next_index]
        next_layer.apply_operation(InputPruner(
            to_remove, indices.size, next_layer.weights))

        instances.append(instance)

    group.instances = instances
