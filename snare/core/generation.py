from ..wrappers import ModelWrapper

from copy import deepcopy
from tensorflow.python.keras import backend as K
import numpy as np


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

    def infer_training_set(self, dataset):
        (x_train, _), _ = dataset
        current_in = x_train[0:2000]
        for group in self.groups:
            current_in = group.infer_base(current_in)
        print("INFER DATASET FINISHED")

    def eval_groups(self, dataset, expected, epsilon, **kwargs):
        assert len(self.group_best) == 0
        K.clear_session()

        # self.infer_training_set(dataset)

        result = self.base

        for group in reversed(self.groups):
            if not group.instances:
                continue

            print()
            print("------------------------------------------------")
            print("Process group with layer='", group.main_layer, "'", sep="")
            print("------------------------------------------------")
            print()

            group.full_wrapper = result

            p_update = group.eval_full(
                dataset, expected, epsilon, self.path, **kwargs)

            # p_update = group.eval(
            #     dataset, expected, epsilon, self.path, **kwargs)

            if p_update < 2:
                result.update(group.result)

            print()
            print("------------------------------------------------")
            print("Finished group with layer='", group.main_layer, "'", sep="")
            print("------------------------------------------------")
            print()



        m = result.to_model()
        (x_train, y_train), (x_test, y_test) = dataset
        test_score = m.evaluate(x_test, y_test)
        # if test_score[1] > expected:
        #     expected = test_score[1]

        # print("Retrain another 5 epochs")

        # m = result.to_model()
        # m.compile(**kwargs)
        # m.evaluate(x_test, y_test)

        # m.fit(x=x_train, y=y_train,
        #       epochs=5, batch_size=128,
        #       validation_data=(x_test, y_test), verbose=1)

        # if hist.history['val_acc'][-1] > expected - 0.5 * epsilon:
        #     base = ModelWrapper.from_model(model, self.path,
        #                                    "result" + str(group.id))
        #     self.result = base
        # else:
        #     print("Retrain does not restore enough!")
        #     group.best_index += 1

        # print("Non-zero:",
        #       np.count_nonzero(m.get_layer("dense").get_weights()[0]))
        # base = ModelWrapper.from_model(m, self.path,
        #                                "result" + str(group.id))

        # self.result = base
        return p_update
