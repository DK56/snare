import os
import json
import sys
from copy import deepcopy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from .generation import Generation
from .group import Group
from .model_wrapper import ModelWrapper
from .operation import prune_random_connections, prune_low_activation_neurons, prune_low_gradient_connections, prune_low_magnitude_connections, prune_low_gradient_neurons, prune_low_magnitude_neurons, prune_random_neurons, InputPruner, NeuronPruner


class Generator():
    IMPORTANT = ['conv2d', 'conv2d_1', 'conv1d', 'conv1d_1',
                 'conv1d_2', 'conv1d_3', 'conv1d_4', 'dense', 'dense_1']

    def __init__(self, model: Sequential, compile_args, tmp_path):
        self.model = model
        self.compile_args = compile_args
        self.tmp_path = tmp_path
        # self.gens = []
        self.current_gen = -1
        self.layer_status = {}

    def get_current_gen(self):
        # return self.gens[self.current_gen]
        return self.gen

    def calculate_layer_score(self):
        base = self.get_current_gen().result
        new_status = self.layer_status

        # (% percentage, % filters, % params, % flops)
        if self.current_gen == 0:
            for layer in base.layers:
                new_status[layer] = (0, 0, 0, 0)

            for layer in base.layers:
                if layer.is_important():
                    new_status[layer] = (32, 0, 0, 0)
                if(layer.name == "dense_2"):
                    new_status[layer] = (0, 0, 0, 0)

        m_neurons = 0
        m_params = 0
        m_flops = 1
        for layer in base.layers:
            m_neurons += layer.neurons
            m_params += layer.params
            m_flops += layer.flops

        print("FLOPS:", m_flops)
        print("Params:", m_params)

        next_flops = 0
        next_params = 0

        for layer in reversed(base.layers):
            # if(layer.name == "conv1d"):
            #     new_status[layer] = (
            #         1, 1, 1, 1)
            # else:
            #     new_status[layer] = (
            #         0, 0, 0, 0)
            # continue

            p, _, _, _ = new_status[layer]

            skip = False
            if p == 0:
                skip = True
            elif p <= 2:
                new_status[layer] = (0, 0, 0, 0)
                skip = True
            elif (p * layer.neurons) / 100 == (4 * layer.neurons) / 100:
                p = 4

            if skip:
                if layer.is_important():
                    next_flops = layer.neurons
                    next_params = layer.params
                continue

            neuron_score = layer.neurons / m_neurons

            params_score = (layer.params + next_params) / (2 * m_params)
            flops_score = (layer.flops + next_flops) / (2 * m_flops)

            new_status[layer] = (p, neuron_score, params_score, flops_score)

            next_flops = layer.neurons
            next_params = layer.params

        self.layer_status = new_status

    def get_best_layer(self):
        base = self.get_current_gen().result
        best_value = 0
        for layer in base.layers:
            p, n_score, p_score, f_score = self.layer_status[layer]
            score = p * (n_score + p_score + f_score)
            if score > best_value:
                best = layer
                best_value = score

        assert(best_value > 0)
        return best

    def prepare(self, dataset):
        assert os.path.isdir(self.tmp_path)
        assert len(os.listdir(self.tmp_path)) == 0

        self.dataset = dataset

        # Build gen_0
        gen_path = os.path.join(self.tmp_path, 'gen_0')
        os.mkdir(gen_path)
        base = ModelWrapper.from_model(self.model, self.compile_args, gen_path)

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

        self.gen = Generation(0, base, gen_path)
        # self.gens.append(Generation(0, base, gen_path))
        self.current_gen = 0

    def build_next_gen(self, pruning_layer=None):
        K.clear_session()
        assert self.current_gen >= 0
        assert os.path.isdir(self.tmp_path)

        # base = self.gens[self.current_gen].result
        base = self.get_current_gen().result

        for layer in base.layers:
            p, n_score, p_score, f_score = self.layer_status[layer]
            if layer.is_important():
                print(layer.name, ": ", p * (n_score + p_score +
                                             f_score) / 100., p, n_score, p_score, f_score)
            else:
                assert(p == 0)

        best = self.get_best_layer()
        current_gen = self.current_gen + 1

        gen_path = os.path.join(self.tmp_path, 'gen_' + str(current_gen))
        assert not os.path.exists(gen_path)
        os.mkdir(gen_path)

        gen = Generation(current_gen, base, gen_path)

        # if current_gen % 2:
        #     groups = Group.create_groups(base, 1, 0)
        # else:
        #     groups = Group.create_groups(base, 2, 1)

        # if best.name in ['conv1d', 'conv1d_2', 'conv1d_4', 'dense_1']:
        # if best.name in ['conv2d', 'dense']:
        # if best.name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block3_conv3', 'block4_conv2', 'block5_conv1', 'block5_conv3', 'dense_1']:
        # if best.name in ['dense_1']:
        #     groups = Group.create_groups(base, 2, 3)
        # else:
        #     groups = Group.create_groups(base, 2, 0)
        # groups = Group.create_groups(base, 1, 0)

        groups = Group.find_layer_groups(best.name, base)

        print()
        print("------------------------------------------------")
        print("Build all groups for generation", current_gen)
        print("------------------------------------------------")
        print()
        print("Best:", best)

        for group in groups:
            gen.add_group(group)
            print(group.main_layer.name)
            if group.main_layer != best:
                continue

            p, _, _, _ = self.layer_status[best]
            if p >= 4:
                percentages = [p / 100., (p / 2) / 100.]
            else:
                percentages = [p / 100.]
            # prune_low_gradient_neurons(group, percentages, self.dataset)
            # prune_low_magnitude_neurons(group, percentages)
            prune_low_gradient_connections(group, percentages, self.dataset)
            # prune_random_connections(group, percentages)
            # prune_random_neurons(group, percentages)

        print()
        print("------------------------------------------------")
        print("Finished building of groups for generation", current_gen)
        print("------------------------------------------------")
        print()

        self.current_gen = current_gen
        # self.gens.append(gen)
        self.gen = gen
        return gen

    def update_status(self, update_value):
        best = self.get_best_layer()
        q, a, b, c = self.layer_status[best]
        self.layer_status[best] = (q >> update_value, a, b, c)

    def has_next(self):
        for status in self.layer_status.values():
            if status[0] != 0:
                return True
        return False

    def get_model_wrapper(self, gen, i):
        assert gen >= 0 and gen < len(self.gens)
        assert i >= 0 and i < len(self.gens[gen])
        return self.gens[gen][i]
