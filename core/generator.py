import os
import json
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
        self.layer_status = {}

    def prepare(self):
        assert os.path.isdir(self.tmp_path)
        assert len(os.listdir(self.tmp_path)) == 0

        # Build gen_0
        gen_path = os.path.join(self.tmp_path, 'gen_0')
        os.mkdir(gen_path)
        splitter = ModelSplitter(self.model)
        layer_dict = splitter.split_layers_to_dir(gen_path)
        weights_dict = splitter.split_weights_to_dir(gen_path)
        order = list(map(lambda layer: layer.name, self.model.layers))
        for layer in order:
            # TODO
            self.layer_status[layer] = 6
        self.layer_status['dense'] = 0
        base = ModelStructure(order, layer_dict, weights_dict)
        self.gens.append(Generation(0, base))
        self.current_gen = 0

    def build_next_gen(self):
        assert self.current_gen >= 0
        assert os.path.isdir(self.tmp_path)
        current_gen = self.current_gen + 1

        gen_path = os.path.join(self.tmp_path, 'gen_' + str(current_gen))
        assert not os.path.exists(gen_path)
        os.mkdir(gen_path)

        base = self.gens[current_gen - 1].result
        gen = Generation(current_gen, base)

        for (i, layer) in enumerate(base.order):
            if self.layer_status[layer] != 6:
                layer_path = os.path.join(gen_path, layer)
                assert not os.path.exists(layer_path)
                os.mkdir(layer_path)

                status = self.layer_status[layer]
                group = []
                weights = base.weights[layer].get()
                percentages = [0.6, 0.3, 0.2, 0.1, 0.02, 0.01]
                for k, p in enumerate(percentages):
                    if status > k:
                        continue
                    to_remove = prune_low_magnitude_neurons(weights, p)
                    next_layer = base.order[i + 1]
                    layer_dict = base.layers.copy()
                    weights_dict = base.weights.copy()
                    op = NeuronPruner(to_remove, weights_dict[layer])
                    with open(layer_dict[layer]) as file:
                        config = json.load(file)

                    config = op.update_config(config)
                    weights_dict[layer] = op

                    config_json = json.dumps(config)
                    config_path = os.path.join(
                        layer_path, "config_" + str(k) + ".json")

                    with open(config_path, 'w+') as f:
                        f.write(config_json)

                    layer_dict[layer] = config_path

                    weights_dict[next_layer] = InputPruner(
                        to_remove, weights_dict[next_layer])

                    order = base.order
                    group.append(ModelStructure(
                        order, layer_dict, weights_dict))
                gen.add_group(group)

        self.current_gen = current_gen
        self.gens.append(gen)

        return gen

    def train_gen(self, dataset, **kwargs):
        gen = self.gens[self.current_gen]
        gen_path = os.path.join(self.tmp_path, 'gen_' + str(self.current_gen))
        gen.train_result(gen_path, dataset, **kwargs)
        self.update_status()

    def update_status(self):
        gen = self.gens[self.current_gen]
        index = gen.group_best[0]
        if index == -1:
            self.layer_status['dense'] = 6
        else:
            self.layer_status['dense'] += index

    def has_next(self):
        for layer_status in self.layer_status.values():
            if layer_status < 6:
                return True
        return False

    def get_model_structure(self, gen, i):
        assert gen >= 0 and gen < len(self.gens)
        assert i >= 0 and i < len(self.gens[gen])
        return self.gens[gen][i]
