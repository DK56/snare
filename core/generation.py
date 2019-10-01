class Generation():

    def __init__(self, number, model_structures, best_pos=-1):
        self.number = number
        self.model_structures = model_structures
        self.best_pos = best_pos

    def get_best(self):
        assert self.best_pos >= 0
        return self.model_structures[self.best_pos]
