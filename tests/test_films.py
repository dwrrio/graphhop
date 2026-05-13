import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_hop import GraphHop

gh = GraphHop()

gh.add_node("Inception", [0, 0, 0, 1])
gh.add_node("Interstellar", [0, 0, 0, 1])
gh.add_node("Dark Knight", [1, 0, 1, 0])
gh.add_node("Superbad", [0, 1, 0, 0])
gh.add_node("21 Jump Street", [0, 1, 0, 0])


gh.add_edge("Inception", "Interstellar")
gh.add_edge("Inception", "Dark Knight")
gh.add_edge("Dark Knight", "Superbad")
gh.add_edge("Superbad", "21 Jump Street")

cue = gh.graph.nodes["Interstellar"]['feature']
result = gh.navigate("Interstellar", cue, steps=2)
print(result)
