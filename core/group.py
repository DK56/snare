import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from .model_wrapper import ModelWrapper


class Group():

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

        wrapper = model_wrapper.copy()

        groups = []

        processable_layers = []
        for i, layer in enumerate(wrapper.layers):
            if layer.is_important():
                processable_layers.append(i)

        start = 0
        end = 0
        current_group_size = 0
        group_number = 0

        max_end = len(wrapper.layers)
        print(processable_layers)

        for i, index in enumerate(processable_layers):
            if index < offset:
                continue

            current_group_size += 1
            if current_group_size == 1:
                main_layer = wrapper.layers[index]

            if current_group_size == min_group_size:
                if i + min_group_size >= len(processable_layers):
                    end = max_end
                else:
                    end = processable_layers[i + 1]

                group = cls(group_number, main_layer, wrapper,
                            ModelWrapper.from_model_wrapper(wrapper,
                                                            start, end))
                group_number += 1
                groups.append(group)

                start = end
                current_group_size = 0
                if end == max_end:
                    break

        return groups

    def infer_base(self, to_infer):
        self.in_data = to_infer
        model = self.base_wrapper.to_model()
        model.compile(loss=losses.mse, optimizer="SGD", metrics=["accuracy"])
        self.out_data = model.predict(to_infer)
        return self.out_data

    def eval(self, dataset, expected, epsilon, path, **kwargs):
        (x_train, y_train), (x_test, y_test) = dataset
        print("Evaluate group", self.id)
        print("Main layer =", self.main_layer)
        for i, instance in enumerate(self.instances):

            model = instance.to_model()
            model.compile(loss=losses.mse, optimizer="SGD",
                          metrics=["accuracy"])

            # diff_dict = {
            #    key: val for (key, val) in tmp.layer_configs.items()
            #    if val != self.base.layer_configs[key]}

            hist = model.fit(x=self.in_data, y=self.out_data,
                             epochs=20, batch_size=128, verbose=1)

            print("Accuracy: " + str(hist.history['acc'][-1]))
            print("Expected: >" + str(0.99))
            if hist.history['acc'][-1] > 0.98:
                print("Found")
                self.result = ModelWrapper.from_model(
                    model, path, "group_" + str(self.id) + "_" + str(i))

                self.best_index = i
                return True

        print("Finished group", self.id, "no improvement")
        self.result = self.base_wrapper
        self.best_index = -1
        return False

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
