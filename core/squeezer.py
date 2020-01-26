from .generator import Generator

import os
from tensorflow.python.keras.models import Sequential


class Squeezer():
    def __init__(self, model: Sequential, compile_args):
        self.model = model
        self.compile_args = compile_args

    def squeeze(self, dataset,
                threshold, main_metric='val_acc', metric_val=None,
                batch_size=128, tmp_path=os.getcwd(), verbose=1) -> Sequential:

        # Set tmp directory to save upcoming models
        assert os.path.isdir(tmp_path)
        tmp_path = os.path.join(tmp_path, 'tmp')
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        model_dir = self.model.name
        if os.path.exists(os.path.join(tmp_path, model_dir)):
            model_dir = model_dir + '_'
            # Find first unique model directory
            i = 1
            while os.path.exists(
                    os.path.join(tmp_path, model_dir + str(i))):
                i += 1
            model_dir = model_dir + str(i)
        model_path = os.path.join(tmp_path, model_dir)
        os.mkdir(model_path)

        if metric_val is None:
            # Evaluate reference metric
            self.model.compile(**self.compile_args)
            _, (x_test, y_test) = dataset
            hist = self.model.evaluate(x_test, y_test, batch_size)
            metric_val = hist[1]

            if verbose > 0:
                print("Reference", main_metric, ":", hist[1])

        generator = Generator(self.model, self.compile_args, model_path)
        # Create first generation
        generator.prepare(dataset, main_metric, metric_val, batch_size)

        # Initial evaluation of all layers
        generator.calculate_layer_score()

        while(generator.has_next()):
            if verbose > 0:
                print()
                print()
                print("------------------------------------------------")
                print("Start generation")
                print("------------------------------------------------")
                print()
                print()
            # Create all potential operations on groups of layers
            gen = generator.build_next_gen()
            # Evaluate all groups
            update_value = gen.eval_groups(dataset, metric_val, threshold)
            generator.update_status(update_value)

            if verbose > 0:
                print()
                print()
                print("------------------------------------------------")
                print("Reference", main_metric, ":", metric_val)
                print("End generation")
                print("------------------------------------------------")
                print()
                print()
            generator.calculate_layer_score()

        return generator.gens[-2].result.to_model()
