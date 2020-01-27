import numpy as np
from abc import ABC, abstractmethod
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras.callbacks import ModelCheckpoint
from .model_wrapper import ModelWrapper
from tensorflow.python.keras import backend as K
import os


class Group():
    state = 0

    def __init__(self, group_index, main_layer, full_wrapper, base_wrapper):
        self.main_layer = main_layer
        self.base_wrapper = base_wrapper
        self.full_wrapper = full_wrapper
        self.instances = []
        self.id = group_index
        self.best_index = -1

    @classmethod
    def _get_processable_layers(cls, wrapper):
        processable_layers = []
        for i, layer in enumerate(wrapper.layers):
            if layer.is_important():
                processable_layers.append(i)
        return processable_layers

    @classmethod
    def find_layer_groups(cls, layername, model_wrapper):
        # TODO split in three separate models
        wrapper = model_wrapper.copy()

        layernames = [layer.name for layer in wrapper.layers]
        main_layer = wrapper.layers[layernames.index(layername)]

        group = cls(0, main_layer, wrapper, wrapper)

        return [group]

    @classmethod
    def create_groups(cls, model_wrapper, min_group_size, offset=0):
        assert min_group_size > 0

        wrapper = model_wrapper.copy()
        groups = []

        processable_layers = Group._get_processable_layers(wrapper)

        start = 0
        end = 0
        current_group_size = 0
        group_number = 0

        max_end = len(wrapper.layers)

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
        model = self.base_wrapper.to_model()
        model.compile(loss=losses.mse, optimizer="SGD", metrics=["accuracy"])
        tmp = model.predict(to_infer)
        if self.instances:
            self.in_data = to_infer
            self.out_data = tmp
        return tmp

    def eval(self, dataset, expected, epsilon, path, **kwargs):
        (x_train, y_train), (x_test, y_test) = dataset
        print("Evaluate group", self.id)
        print("Main layer =", self.main_layer)

        update = 0
        for i, instance in enumerate(self.instances):
            tmp = self.base_wrapper.copy()
            tmp.update(instance)

            model = tmp.to_model()
            model.compile(loss=losses.mse, optimizer="Adam",
                          metrics=["accuracy"])

            # Retrain
            hist = model.fit(x=self.in_data, y=self.out_data,
                             epochs=50, batch_size=128, verbose=1)

            print("Accuracy: " + str(hist.history['acc'][-1]))
            print("Expected: >" + str(0.99))
            for value in hist.history['acc']:
                if value > 0.95:
                    print("Found")
                    tmp.update(ModelWrapper.from_model(
                        model, path, "group_" + str(self.id) + "_" + str(i)))

                    self.result = tmp

                    print("Finished group", self.id, "new model saved")
                    self.best_index = i
                    return update
            update += 1

        print("Finished group", self.id, "no improvement")
        self.result = self.base_wrapper
        self.best_index = -1
        return update

    def eval_full(self, dataset, expected, epsilon, path):
        (x_train, y_train), (x_test, y_test) = dataset
        print("Evaluate group", self.id)
        print("Main layer =", self.main_layer)

        update = 0

        for i, instance in enumerate(self.instances):

            tmp = self.full_wrapper.copy()
            tmp.update(instance)

            model = tmp.to_model()
            tmp_path = os.path.join(path, 'tmp_result.h5')
            checkpoint = ModelCheckpoint(
                tmp_path, monitor='val_acc', verbose=0,
                save_best_only=True, mode='max')

            hist = model.fit(x=x_train, y=y_train,
                             epochs=5, batch_size=128,
                             validation_data=(x_test, y_test),
                             callbacks=[checkpoint], verbose=2)
            
            for value in hist.history['val_acc']:
                if value > expected - epsilon:
                    tmp.update(ModelWrapper.from_model(
                        load_model(tmp_path), tmp.compile_args,
                        path, "group_" + str(self.id)))
                    self.result = tmp

                    print("Finished group", self.id, "new model saved")
                    self.best_index = i
                    return update
            update += 1

        print("Finished group", self.id, "no improvement")
        self.result = self.full_wrapper
        self.best_index = -1
        return update
