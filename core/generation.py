from copy import deepcopy
from .model_wrapper import ModelWrapper


class Generation():

    def __init__(self, number, base):
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

    def train_result(self, path, dataset, **kwargs):
        (x_train, y_train), (x_test, y_test) = dataset
        model = self.result.to_model()
        model.compile(**kwargs)
        hist = model.fit(x=x_train, y=y_train,
                         epochs=10, batch_size=128,
                         validation_data=(x_test, y_test), verbose=1)
        print("Accuracy after result training: " +
              str(hist.history['val_acc'][-1]))
        self.result = ModelWrapper.from_model(model, path, 'result_trained')
        model.summary()

    def eval_groups(self, dataset, expected, epsilon, **kwargs):
        assert len(self.group_best) == 0
        (x_train, y_train), (x_test, y_test) = dataset
        for group in self.groups:
            for i, wrapper in enumerate(group):
                model = wrapper.to_model()
                model.compile(**kwargs)
                diff_dict = {key: val for (key, val) in wrapper.layer_configs.items()
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
