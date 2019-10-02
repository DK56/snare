import os
from tensorflow.python.keras.models import Sequential
from .generation import Generation
from .model_splitter import ModelSplitter
from .model_structure import ModelStructure
from .operation import prune_low_magnitude_neurons, InputPruner, NeuronPruner


class Generator():

    def __init__(self, model: Sequential, tmp_path):
        self.model = model
        self.tmp_path = tmp_path
        self.gens = []
        self.current_gen = -1

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
        self.gens.append(Generation(0, model_structure_init))
        self.current_gen = 0

    def build_next_gen(self):
        assert self.current_gen >= 0
        assert os.path.isdir(self.tmp_path)
        current_gen = self.current_gen + 1

        gen_n_path = os.path.join(self.tmp_path, 'gen_' + current_gen)
        assert not os.path.exists(gen_n_path)
        os.mkdir(gen_n_path)

        self.current_gen = current_gen
        # Build gen_0

    def get_model_structure(self, gen, i):
        assert gen >= 0 and gen < len(self.gens)
        assert i >= 0 and i < len(self.gens[gen])
        return self.gens[gen][i]
