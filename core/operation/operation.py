import numpy as np
from .. import Weightprovider


class Operation(Weightprovider):

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
        w = np.delete(w, self.to_remove, len(w) - 1)
        b = np.delete(b, self.to_remove)
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

    sums = np.sum(w, axis=w.ndim - 1)
    indices = np.argsort(sums)
    return indices[0:percentage * indices.size]
    
    
