import numpy as np
import math
from .model_structure import WeightsProvider


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
        if 'units' in config['config']:
            config['config']['units'] = config['config']['units'] - \
                len(self.to_remove)
        if 'filters' in config['config']:
            config['config']['filters'] = config['config']['filters'] - \
                len(self.to_remove)
        return config


class InputPruner(Operation):

    def __init__(self, to_remove, weights_provider):
        self.to_remove = to_remove
        super().__init__(weights_provider)

    def get(self):
        weights = self.weights_provider.get()
        w = weights[0]
        w = np.delete(w, self.to_remove, w.ndim - 2)
        return [w, weights[1]]


def prune_low_magnitude_neurons(weights, percentage):
    w = weights[0]
    # b = weights[1]

    sums = np.sum(w, axis=w.ndim - 2)
    indices = np.argsort(sums)
    # print(indices.shape)
    # print(indices[0: int(percentage * indices.size)].shape)
    print("Prune " + str(int(percentage * 100)) + "%:" +
          str(len(indices[0: math.ceil(percentage * indices.size)])) + " neurons")
    return indices[0: math.ceil(percentage * indices.size)]
