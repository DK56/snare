import os
import json
import sys
from copy import deepcopy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import backend as K
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

        # TODO
        for layer in base.layers:
            self.layer_status[layer] = 6
        for layer in base.layers:
            if layer.name in ['conv1d_1', 'conv1d_2', 'conv1d_3', 'conv1d_4',
                              'dense', 'dense_1']:
                self.layer_status[layer] = 0

        self.gens.append(Generation(0, base, gen_path))
        self.current_gen = 0

    def build_next_gen(self):
        K.clear_session()
        assert self.current_gen >= 0
        assert os.path.isdir(self.tmp_path)
        current_gen = self.current_gen + 1

        gen_path = os.path.join(self.tmp_path, 'gen_' + str(current_gen))
        assert not os.path.exists(gen_path)
        os.mkdir(gen_path)

        base = self.gens[current_gen - 1].result
        gen = Generation(current_gen, base, gen_path)

        groups = Group.create_groups(base, 2, 0)
        if current_gen % 2:
            groups = Group.create_groups(base, 2, 0)
        else:
            groups = Group.create_groups(base, 2, 1)

        print()
        print("------------------------------------------------")
        print("Build all groups for generation", current_gen)
        print("------------------------------------------------")
        print()

        for group in groups:
            gen.add_group(group)

            status = self.layer_status[group.main_layer]
            if status == 6:
                continue

            percentages = [0.6, 0.3, 0.2, 0.1, 0.02, 0.01]
            percentages = [p for i, p in enumerate(percentages) if i >= status]
            prune_low_magnitude_neurons(group, percentages)

        print()
        print("------------------------------------------------")
        print("Finished building of groups for generation", current_gen)
        print("------------------------------------------------")
        print()

        self.current_gen = current_gen
        self.gens.append(gen)
        return gen

    def update_status(self):
        gen = self.gens[self.current_gen]

        for group in gen.groups:
            index = group.best_index
            if index == -1:
                self.layer_status[group.main_layer] = 6
            else:
                self.layer_status[group.main_layer] += index

    def has_next(self):
        for layer_status in self.layer_status.values():
            if layer_status < 6:
                return True
        return False

    def get_model_wrapper(self, gen, i):
        assert gen >= 0 and gen < len(self.gens)
        assert i >= 0 and i < len(self.gens[gen])
        return self.gens[gen][i]
