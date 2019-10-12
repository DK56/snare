import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from .model_wrapper import ModelWrapper


class Group():
    IMPORTANT_LAYERS = ['Conv2D', 'Dense']

    def __init__(self, group_index, main_layer, full_wrapper, base_wrapper):
        self.main_layer = main_layer
        self.base_wrapper = base_wrapper
        self.full_wrapper = full_wrapper
        self.instances = []
        self.id = group_index
        self.best_index = -1

    @classmethod
    def create_groups(cls, model_wrapper, min_group_size, offset=0):
        assert min_group_size > 0
        # TODO Needs to be an deepcopy
        wrapper = model_wrapper

        groups = []

        processable_layers = []
        for i, layer_name in enumerate(wrapper.order):
            layer_config = wrapper.layer_configs[layer_name]

            if layer_config['class_name'] in ['Conv2D', 'Dense']:
                processable_layers.append(i)

        start = 0
        end = 0
        current_group_size = 0
        group_number = 0

        max_end = len(wrapper.order)
        print(processable_layers)

        for i, index in enumerate(processable_layers):
            if index < offset:
                continue

            current_group_size += 1
            if current_group_size == 1:
                main_layer = wrapper.order[index]

            if current_group_size == min_group_size:
                if i + min_group_size >= len(processable_layers):
                    end = max_end
                else:
                    end = processable_layers[i + 1]

                group = cls(group_number, main_layer, wrapper, ModelWrapper.from_model_wrapper(
                    wrapper, start, end))
                group_number += 1
                groups.append(group)

                start = end
                current_group_size = 0
                if end == max_end:
                    break

        return groups

    def eval(self, dataset, train, expected, epsilon, **kwargs):
        (x_train, y_train), (x_test, y_test) = dataset
        for i, wrapper in enumerate(self.instances):
            model = wrapper.to_model()
            model.compile(**kwargs)
            diff_dict = {
                key: val for (key, val) in wrapper.layer_configs.items()
                if val != self.base.layer_configs[key]}

            hist = model.fit(x=x_train, y=y_train,
                             epochs=5, batch_size=128,
                             validation_data=(x_test, y_test), verbose=1)

            print("Accuracy: " + str(hist.history['val_acc'][-1]))
            print("Expected: >" + str(expected - epsilon))
            if hist.history['val_acc'][-1] > expected - epsilon:
                print("Found")
                diff_dict = {key: val for key, val in wrapper.layer_configs.items()
                             if val != self.base.layer_configs[key]}
                self.result.layer_configs.update(diff_dict)
                diff_dict = {key: val for key, val in wrapper.layer_weights.items()
                             if val != self.base.layer_weights[key]}
                self.result.layer_weights.update(diff_dict)
                self.group_best.append(i)

                break
        self.group_best.append(-1)

    def eval_full(self, dataset, expected, epsilon, path, **kwargs):
        (x_train, y_train), (x_test, y_test) = dataset
        print("Evaluate group", self.id)
        print("Main layer =", self.main_layer)
        for i, instance in enumerate(self.instances):

            tmp = self.full_wrapper.copy()
            tmp.update(instance)

            model = tmp.to_model()
            model.compile(**kwargs)

            # diff_dict = {
            #    key: val for (key, val) in tmp.layer_configs.items()
            #    if val != self.base.layer_configs[key]}

            hist = model.fit(x=x_train, y=y_train,
                             epochs=5, batch_size=128,
                             validation_data=(x_test, y_test), verbose=1)

            print("Accuracy: " + str(hist.history['val_acc'][-1]))
            print("Expected: >" + str(expected - epsilon))
            if hist.history['val_acc'][-1] > expected - epsilon:
                print("Found")
                self.result = ModelWrapper.from_model(
                    model, path, "group_" + str(self.id))

                print("Finished group", self.id, "new model saved")
                self.best_index = i
                return True

        print("Finished group", self.id, "no improvement")
        self.result = self.full_wrapper
        self.best_index = -1
        return False
