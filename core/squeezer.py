import os
from tensorflow.python.keras.models import Sequential
from . import Generator
from .model_wrapper import LayerWrapper
from .group import Group
from tensorflow.python.keras import losses
import tensorflow as tf
from tensorflow.python.keras.datasets import mnist
from tensorflow.python import keras
import tensorflow.keras.backend as K
import numpy as np


class Squeezer():
    def __init__(self, model: Sequential, compile_args, tmp_path=os.getcwd()):
        self.model = model
        self.compile_args = compile_args
        assert os.path.isdir(tmp_path)
        self.tmp_path = os.path.join(tmp_path, 'tmp')
        s = 0
        for layer in model.layers:
            flops = LayerWrapper.get_flops(layer)
            print(layer.name, flops)
            s += flops

        print("FLOPS:", s)

    def squeeze(self, dataset,
                threshold, main_metric='val_acc', metric_val=None,
                batch_size=32) -> Sequential:

        if metric_val is None:
            self.model.compile(**self.compile_args)
            _, (x_test, y_test) = dataset
            self.model.evaluate(x_test, y_test, batch_size)

        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
        model_dir = self.model.name
        if os.path.exists(os.path.join(self.tmp_path, model_dir)):
            model_dir = model_dir + '_'
            i = 1
            while os.path.exists(
                    os.path.join(self.tmp_path, model_dir + str(i))):
                i += 1
            model_dir = model_dir + str(i)
        model_path = os.path.join(self.tmp_path, model_dir)
        os.mkdir(model_path)

        generator = Generator(self.model, self.compile_args, model_path)
        generator.prepare(dataset)
        data = [0.9084]
        w = self.model.count_params()
        generator.calculate_layer_score()

        while(generator.has_next()):
            print()
            print()
            print("------------------------------------------------")
            print("Start generation")
            print("------------------------------------------------")
            print()
            print()
            gen = generator.build_next_gen()
            # update_value = gen.eval_groups(dataset, 0.9083, threshold,
            # update_value = gen.eval_groups(dataset, 0.99, threshold)
            update_value = gen.eval_groups(dataset, 0.7009, threshold)

            # gen.train_result(dataset,
            #                  loss=losses.categorical_crossentropy,
            #                  optimizer="SGD", metrics=["accuracy"])

            generator.update_status(update_value)

            log = False
            if log:
                m = gen.result.to_model()
                _, (x_test, y_test) = dataset
                score = m.evaluate(x_test, y_test)
                data.append(score[1])
                print("[", end="")
                for i, d in enumerate(data):
                    print(
                        "(" + str(i) + "," + "{0:.4f}".format(d) + ") ", end="")
                print("]", end="")

            if generator.has_next():
                model = generator.gens[-1].result.to_model()
                model.summary()

            print()
            print()
            print("------------------------------------------------")
            print("End generation")
            print("------------------------------------------------")
            print()
            print()
            generator.calculate_layer_score()

        # best_model_structure = generator.get_model_structure(0, 0)
        # best_model_structure.to_model()
        return generator.gens[-2].result.to_model()
