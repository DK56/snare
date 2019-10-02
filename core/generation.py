from .model_structure import ModelStructure


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
        self.group_best.append(-1)

    def build_group_element(self, group_number, pos):
        order = self.base.order.copy()
        layers = self.base.layers.copy()
        weights = self.base.weights.copy()

        layers.update(self.groups[group_number][pos].layers)
        weights.update(self.groups[group_number][pos].weights)

        return ModelStructure(order, layers, weights)

    def build_result(self):
        assert len(self.groups) == len(self.group_best)
        order = self.base.order.copy()
        layers = self.base.layers.copy()
        weights = self.base.weights.copy()

        for i, group in enumerate(self.groups):
            best = self.group_best[i]
            assert best >= 0 and best < len(group)

            layers.update(group[best].layers)
            weights.update(group[best].weights)

        return ModelStructure(order, layers, weights)

    def eval_groups(self, dataset, expected, epsilon, **kwargs):
        (x_train, y_train), (x_test, y_test) = dataset
        for group in self.groups:
            for structure in group:
                model = structure.to_model()
                model.compile(**kwargs)
                hist = model.fit(x=x_train, y=y_train,
                                 epochs=5, batch_size=128,
                                 validation_data=(x_test, y_test), verbose=1)
                print(hist.history['acc'][-1])
