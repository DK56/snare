from ..wrappers import ModelWrapper

from tensorflow.python.keras import backend as K


class Generation():
    """A generation of SNARE which contains groups of layers.
    Each group can be evaluated separately and retrained.
    """

    def __init__(self, number, base, path):
        """
        Args:
          number: The current generation number.
          base: The base group of all following groups.
          path: Directory-path to save all ModelWrappers of this gen.
        """
        self.path = path
        self.number = number
        self.base = base
        self.groups = []
        self.group_best = []
        self.group_results = []
        self.result = base

    def add_group(self, group):
        """ Adds a group to this gen.

        Args:
          group: The group to add.
        """
        self.groups.append(group)

    def train_result(self, dataset, compile_args):
        """ Trains result of this generation.

        Args:
          dataset: The trainings-dataset.
          compile_args: All keras-compile arguments to compile this model
        """
        (x_train, y_train), (x_test, y_test) = dataset
        model = self.result.to_model()
        model.compile(**compile_args)
        hist = model.fit(x=x_train, y=y_train,
                         epochs=10, batch_size=128,
                         validation_data=(x_test, y_test), verbose=1)
        print("Accuracy after result training: " +
              str(hist.history['val_acc'][-1]))
        self.result = ModelWrapper.from_model(
            model, self.path, 'result_trained')
        model.summary()

    def infer_training_set(self, dataset):
        """ Creates necessary trainings-data forall groups. Starting from the input,
        it traverses all groups to infer input-data for subsequent groups.

        Args:
          dataset: The trainings-dataset.
        """
        (x_train, _), _ = dataset
        # Only infer 2000 examples to reduce space consumption
        current_in = x_train[0:2000]
        for group in self.groups:
            current_in = group.infer_base(current_in)

    def eval_groups(self, dataset, expected, epsilon):
        """ Evaluates all groups and thus this generation. Currently, there is only 
        one pruned group in each generation.

        Args:
          dataset: The trainings-dataset.
          expected: The reference value of the evaluation metric.
          epsilon: The accepted change of the evaluation metric.
          compile_args: All keras-compile arguments to compile a group
        """
        assert len(self.group_best) == 0

        # Clean all TensorFlow graphs
        # Time-consuming, but necessary to prevent out-of-memory failures
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

            # Evaluate a group
            p_update = group.eval_full(
                dataset, expected, epsilon, self.path)

            if p_update < 2:
                # Apply change of successful pruning operation
                result.update(group.result)

            print()
            print("------------------------------------------------")
            print("Finished group with layer='", group.main_layer, "'", sep="")
            print("------------------------------------------------")
            print()

        # Evaluate resulting model
        m = result.to_model()
        (x_train, y_train), (x_test, y_test) = dataset
        test_score = m.evaluate(x_test, y_test)

        # self.result = base
        return p_update, test_score
