import os
from tensorflow.python.keras.models import Sequential
from .model_splitter import ModelSplitter
from .model_structure import ModelStructure


class Generator():

    def __init__(self, model: Sequential, tmp_path):
        self.model = model
        self.tmp_path = tmp_path
        self.gens = []

    def prepare(self):
        assert os.path.isdir(self.tmp_path)
        assert len(os.listdir(self.tmp_path)) == 0

        # Build gen_0
        gen_0_path = os.path.join(self.tmp_path, 'gen_0')
        os.mkdir(gen_0_path)
        splitter = ModelSplitter(self.model)
        layer_dict = splitter.split_layers_to_dir(gen_0_path)
        weights_dict = splitter.split_weights_to_dir(gen_0_path)
        order = map(lambda layer: layer.name, self.model.layers)
        model_structure_init = ModelStructure(order, layer_dict, weights_dict)
        self.gens.append([model_structure_init])
