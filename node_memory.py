import numpy as np

class NodeMemory:
  def __init__(self, dim):
    self.dim = dim
    self.w = np.zeros((dim, dim))

    self.patterns = []


  def store(self, vector):
    vector = np.array(vector)
    self.patterns.append(vector)

    self.w += np.outer(vector, vecotr)
    np.fill_diagonal(self.w, 0)


  def recall(self, cue):
    return np.sign(self.w @ cue)
