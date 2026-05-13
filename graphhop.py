import numpy as np
import matplotlib.pyplot as plt

class HopfieldMemory:
    def __init__(self, n_neurons):
        self.n = n_neurons
        self.w = np.zeros((n_neurons, n_neurons))


    def train(self, patterns):
        for p in patterns:
            self.w += np.outer(p, p)
        
        np.fill_diagonal(self.w, 0)
        self.w /= len(patterns)
        

    def recall(self, x, steps = 5):
        x = x.copy()

        for _ in range(steps):
            x = np.sign(self.w @ x)
            x[x == 0] = 1

        return x


    def energy(self, x):
        return -.5 * (x @ self.w @ x)
