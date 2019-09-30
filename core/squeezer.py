import os
from tensorflow.python.keras.models import Sequential
from . import Generator


class Squeezer():
    def __init__(self, model: Sequential, tmp_path=os.getcwd()):
        self.model = model
        assert os.path.isdir(tmp_path)
        self.tmp_path = os.path.join(tmp_path, 'tmp')

    def squeeze(self) -> Sequential:
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
        model_dir = self.model.name
        if os.path.exists(os.path.join(self.tmp_path, model_dir)):
            model_dir = model_dir + '_'
            i = 1
            while os.path.exists(os.path.join(self.tmp_path, model_dir + str(i))):
                i += 1
            model_dir = model_dir + str(i)
        model_path = os.path.join(self.tmp_path, model_dir)
        os.mkdir(model_path)

        generator = Generator(self.model, model_path)
        generator.prepare()

        best_model_structure = generator.get_model_structure(0, 0)
        return best_model_structure.to_model()
