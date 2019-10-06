import os
import json
from copy import deepcopy
from tensorflow.python.keras.models import Sequential
from .generation import Generation
from .group import Group
from .model_wrapper import ModelWrapper
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
        base = ModelWrapper.from_model(self.model, gen_path)
        for layer in base.order:
            # TODO
            self.layer_status[layer] = 6
        self.layer_status['dense'] = 0
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

                groups = Group.create_groups(base, 2, 1)
                for group in groups:
                    print(group.order)

                status = self.layer_status[layer]
                group = []
                weights = base.layer_weights[layer].get()
                percentages = [0.6, 0.3, 0.2, 0.1, 0.02, 0.01]
                for k, p in enumerate(percentages):
                    if status > k:
                        continue
                    to_remove = prune_low_magnitude_neurons(weights, p)
                    next_layer = base.order[i + 1]
                    layer_configs = deepcopy(base.layer_configs)
                    layer_weights = base.layer_weights.copy()

                    op = NeuronPruner(to_remove, layer_weights[layer])
                    layer_weights[layer] = op

                    config = op.update_config(layer_configs[layer])
                    layer_configs[layer] = config

                    layer_weights[next_layer] = InputPruner(
                        to_remove, layer_weights[next_layer])

                    order = base.order
                    group.append(ModelWrapper(
                        order, layer_configs, layer_weights))
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

    def get_model_wrapper(self, gen, i):
        assert gen >= 0 and gen < len(self.gens)
        assert i >= 0 and i < len(self.gens[gen])
        return self.gens[gen][i]
