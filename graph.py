import numpy as np

# assuming undirected, for now 
# node label start at zero, they are the same indexes 
class Graph:
	def __init__(self, size):	
		self.adj = np.zeros((size or 1, size or 1))
		num_rows = self.adj.shape[0] 
		self.nodes = num_rows
		self.edges = 0 
		self.update_attributes()

	def add_node(self):
		# will add a new row & column of zeroes to the adjacency matrix 
		num_rows = self.adj.shape[0] 
		b = np.zeros((num_rows + 1, num_rows + 1))
		b[:num_rows, :num_rows] = self.adj
		self.adj = b
		self.nodes += 1
		self.update_attributes()

	def add_edge(self, nodes):
		if len(nodes) == 2 and nodes[0] >= 0 and nodes[1] >= 0:
			self.adj[nodes[0], nodes[1]] = 1
			#to maintain symmettry
			self.adj[nodes[1], nodes[0]] = 1
			self.edges += 1 
			self.update_attributes()

	def has_cycle(self):
		if self.nodes < 3:
			return false


	def update_attributes(self):
		self.degree = self.adj.sum(axis = 0)


g = Graph(5)
g.add_node()
g.add_node()
g.add_edge((2, 1))
g.add_edge((3, 1))
g.add_edge((5, 1))
g.add_edge((0, 6))
print('degree of node 1 :', g.degree[1])
print(g.nodes)
print(g.edges)
print(g.adj)