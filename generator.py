import numpy as np
from graph import Graph as g

class Generator:
	def generate_tree(self):
		g = Graph(7)
		g.add_edge((0, 1))
		g.add_edge((0, 2))
		g.add_edge(1, 3)
		g.add_edge(1, 4)
		g.add_edge(1, 5)
		g.add_edge(1, 6)
		return g
