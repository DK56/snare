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

    def calculate_layer_score(self):
        base = self.gens[self.current_gen].result
        new_status = self.layer_status

        # (% percentage, % filters, % params, % flops)
        if self.current_gen == 0:
            for layer in base.layers:
                new_status[layer] = (0, 0, 0, 0)

            for layer in base.layers:
                if layer.name in Generator.IMPORTANT:
                    new_status[layer] = (64, 0, 0, 0)

        m_neurons = 0
        m_params = 0
        m_flops = 0
        for layer in base.layers:
            m_neurons += layer.neurons
            m_params += layer.params
            m_flops += layer.flops

        for layer in base.layers:
            p, _, _, _ = new_status[layer]
            if p == 0:
                continue
            if p <= 2:
                new_status[layer] = (0, 0, 0, 0)

            neuron_score = layer.neurons / m_neurons
            params_score = layer.params / m_params
            flops_score = layer.flops / m_flops

            new_status[layer] = (p, neuron_score, params_score, flops_score)

    def prepare(self, dataset):
        assert os.path.isdir(self.tmp_path)
        assert len(os.listdir(self.tmp_path)) == 0

        self.dataset = dataset

        # Build gen_0
        gen_path = os.path.join(self.tmp_path, 'gen_0')
        os.mkdir(gen_path)
        base = ModelWrapper.from_model(self.model, gen_path)

        # TODO
        # for layer in base.layers:
        #     self.layer_status[layer] = 4
        # for layer in base.layers:
        #     # if layer.name in ['conv1d_4']:
        #     #     self.layer_status[layer] = 3
        #     if layer.name in ['conv2d', 'conv2d_1', 'dense', 'dense_1', 'dense_2']:
        #         self.layer_status[layer] = 0
        #     if layer.name in ['conv2d_2']:
        #         self.layer_status[layer] = 3
        #     if layer.name in ['conv1d_4', 'dense']:
        #         self.layer_status[layer] = 0
        #     if layer.name in ['conv1d_3', 'conv1d_2', 'conv1d_1', 'dense_1']:
        #         self.layer_status[layer] = 1
        #     if layer.name in ['conv1d']:
        #         self.layer_status[layer] = 2

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
            if status == 4:
                continue

            percentages = [0.6, 0.3, 0.2, 0.001]
            percentages = [p for i, p in enumerate(percentages) if i >= status]
            prune_low_gradient_neurons(group, percentages, self.dataset,
                                       loss=losses.categorical_crossentropy, optimizer="SGD", metrics=["accuracy"])
            # prune_low_magnitude_neurons(group, percentages)
            # prune_random_neurons(group, percentages)

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
                self.layer_status[group.main_layer] = 4
            else:
                self.layer_status[group.main_layer] += index

    def has_next(self):
        for layer_status in self.layer_status.values():
            if layer_status < 4:
                return True
        return False

    def get_model_wrapper(self, gen, i):
        assert gen >= 0 and gen < len(self.gens)
        assert i >= 0 and i < len(self.gens[gen])
        return self.gens[gen][i]
