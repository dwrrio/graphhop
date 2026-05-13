import networkx as nx

class GraphHop:
  def __init__(self):
    self.graph = nx.Graph()
    self.memories = {}  # node_id <-> NodeMemory


  def add_node(self, node_id, feature_vector, feature_dim = None):
    self.graph.add_node(node_id, feature=feature_vector)
    if feature_dim is None:
      feature_dim = len(feature_vector)
