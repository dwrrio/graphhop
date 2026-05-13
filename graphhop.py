import networkx as nx
from node_memory import *

class GraphHop:
  def __init__(self):
    self.graph = nx.Graph()
    self.memories = {}  # node_id <-> NodeMemory


  def add_node(self, node_id, feature_vector, feature_dim = None):
    self.graph.add_node(node_id, feature=feature_vector)
    if feature_dim is None:
      feature_dim = len(feature_vector)
    self.memories[node_id] = NodeMemory(feature_dim)


  def add_edge(self, node1, node2):
    feat1 = self.graph.nodes[node1]['feature']
    feat2 = self.graph.nodes[node2]['feature']

    self.memories[node1].store(feat2)
    self.memories[node2].store(feat1)


  def navigate(self, start_node, cue, steps = 3):
    path = [start_node]
    current = start_node
    current_cue = cue

    for _ in range(steps):
      recalled = self.memories[current].recall(current_cue)
      neighbors = list(self.graph.neighbors(current))

      if not neighbors:
        break

      best_neighbor = None
      best_similarity = -float('inf')

      for neighbor in neighbors:
        neighbor_feat = self.graph.nodes[neighbor]['feature']
        similarity = np.dot(neighbor_feat, recalled) / (np.linalg.norm(neighbor_feat) * np.linalg.norm(recalled))

        if similarity > best_similarity:
          best_similarity = similarity
          best_neighbor = neighbor

      path.append(best_neighbor)
      current = best_neighbor
      current_cue = recalled

    return path
