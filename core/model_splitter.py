import os
import json
import numpy as np
from .model_structure import FileWeights
from tensorflow.python.keras.models import Model


class ModelSplitter():

    def __init__(self, model: Model):
        self.model = model

    def split_layers_to_dir(self, path):
        assert os.path.exists(path)
        assert os.path.isdir(path)

        layer_configs = {}
        config_file = 'config.json'

        for layer in self.model.layers:
            layer_dir = os.path.join(path, layer.name)
            if not os.path.exists(layer_dir):
                os.mkdir(layer_dir)

            config_path = os.path.join(layer_dir, config_file)
            assert not os.path.exists(config_path)

            config = {'class_name': layer.__class__.__name__,
                      'config': layer.get_config()}
            config_json = json.dumps(config)

            with open(config_path, 'w+') as f:
                f.write(config_json)

            layer_configs[layer.name] = config_path

        return layer_configs

    def split_weights_to_dir(self, path):
        assert os.path.exists(path)
        assert os.path.isdir(path)

        layer_weights = {}
        weights_file = 'weights.npy'

        for layer in self.model.layers:
            layer_dir = os.path.join(path, layer.name)
            if not os.path.exists(layer_dir):
                os.mkdir(layer_dir)

            w = FileWeights(os.path.join(layer_dir, weights_file))
            w.save(layer.get_weights())
            layer_weights[layer.name] = w

        return layer_weights
