import numpy as np

class ART1:
    def __init__(self, input_size, vigilance=0.8):
        self.input_size = input_size
        self.vigilance = vigilance
        self.weights = []
        self.num_categories = 0

    def complement_code(self, input_vector):
        return np.concatenate([input_vector, 1 - input_vector])

    def match(self, input_vector, weight):
        return np.sum(np.minimum(input_vector, weight)) / np.sum(input_vector)

    def train(self, inputs):
        inputs = [self.complement_code(x) for x in inputs]
        for x in inputs:
            matched = False
            for i, w in enumerate(self.weights):
                if self.match(x, w) >= self.vigilance:
                    self.weights[i] = np.minimum(w, x)
                    matched = True
                    break
            if not matched:
                self.weights.append(x)
                self.num_categories += 1

    def predict(self, input_vector):
        x = self.complement_code(input_vector)
        for i, w in enumerate(self.weights):
            if self.match(x, w) >= self.vigilance:
                return i
        return -1

X = np.array([
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 0]
])

art = ART1(input_size=4, vigilance=0.8)
art.train(X)

for i, x in enumerate(X):
    print(f"Pattern {x} belongs to category {art.predict(x)}")
