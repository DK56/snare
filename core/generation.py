from copy import deepcopy
from .model_wrapper import ModelWrapper


class Generation():

    def __init__(self, number, base, path):
        self.path = path
        self.number = number
        self.base = base
        self.groups = []
        self.group_best = []
        self.group_results = []
        self.result = base

    def add_group(self, group):
        self.groups.append(group)

    def build_group_element(self, group_number, pos):
        order = self.base.order.copy()
        layer_configs = deepcopy(self.base.layer_configs)
        layer_weights = self.base.layer_weights.copy()

        layer_configs.update(self.groups[group_number][pos].layer_configs)
        layer_weights.update(self.groups[group_number][pos].layer_weights)

        return ModelWrapper(order, layer_configs, weights)

    def build_result(self):
        assert len(self.groups) == len(self.group_best)
        order = self.base.order.copy()
        layer_configs = self.base.layer_configs.copy()
        layer_weights = self.base.layer_weights.copy()

        for i, group in enumerate(self.groups):
            best = self.group_best[i]
            assert best >= 0 and best < len(group)

            layer_configs.update(group[best].layer_configs)
            layer_weights.update(group[best].layer_weights)

        return ModelWrapper(order, layer_configs, layer_weights)

    def train_result(self, dataset, **kwargs):
        (x_train, y_train), (x_test, y_test) = dataset
        model = self.result.to_model()
        model.compile(**kwargs)
        hist = model.fit(x=x_train, y=y_train,
                         epochs=10, batch_size=128,
                         validation_data=(x_test, y_test), verbose=1)
        print("Accuracy after result training: " +
              str(hist.history['val_acc'][-1]))
        self.result = ModelWrapper.from_model(
            model, self.path, 'result_trained')
        model.summary()

    def eval_groups(self, dataset, expected, epsilon, **kwargs):
        assert len(self.group_best) == 0
        (x_train, y_train), (x_test, y_test) = dataset

        base = self.base

        for group in reversed(self.groups):

            print()
            print("------------------------------------------------")
            print("Process group with layer='" + group.main_layer + "'")
            print("------------------------------------------------")
            print()
            group.full_wrapper = base

            group.eval_full(dataset, expected, epsilon, self.path, **kwargs)

            model = group.result.to_model()
            model.compile(**kwargs)

            print("Retrain another 5 epochs")

            model.fit(x=x_train, y=y_train,
                      epochs=5, batch_size=128,
                      validation_data=(x_test, y_test), verbose=1)

            base = ModelWrapper.from_model(model, self.path,
                                           "result" + str(group.id))
            self.result = base

            print()
            print("------------------------------------------------")
            print("Finished group with layer='" + group.main_layer + "'")
            print("------------------------------------------------")
            print()
