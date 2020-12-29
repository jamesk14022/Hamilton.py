import numpy as np
import generator as gen
import visualise as vis
import attributes as att
from tabulate import tabulate

np.set_printoptions(precision=3)


class Graph:
	def __init__(self, size, names = None):	
		self.edges = {}
		self.node_names = {}
		self.nodes = 0
		self.degree = 0
		self.adj = np.zeros((size or 0, size or 0))

		self.custom_node_names = (names is not None) 

		if not self.custom_node_names:
			for x in range(0, size):
				self.node_names[x] = x
		else:
			if len(names) == size:
				for x in range(0, size):
					self.node_names[names[x]] = x

		num_rows = self.adj.shape[0] 
		self.update_attributes()

	def add_nodes(self, size, names = None):
		if names is None:
			if(self.custom_node_names == True):
				print('You shoud have used custom names!')
			else:
				num_rows = self.adj.shape[0] 
				b = np.zeros((num_rows + size, num_rows + size))
				b[:num_rows, :num_rows] = self.adj
				self.adj = b
				# update naming dictionary with index -> index
				for x in range(self.nodes, self.nodes + size):
					self.node_names[x] = x

		elif self.custom_node_names == False:
			print('You shouldn\'t have passed custom names!')

		elif type(names[0]) != type(next(iter(self.node_names.values()))):
			print('Name schemes don\'t match!')

		elif len(names) != size:
			print('You didn\'t pass the right number of names!')

		else:
			# first add rows to adjacency matrix
			num_rows = self.adj.shape[0] 
			b = np.zeros((num_rows + size, num_rows + size))
			b[:num_rows, :num_rows] = self.adj
			self.adj = b
			# update naming dictionary 
			for x in range(self.nodes, self.nodes + size):
				self.node_names[names[x - self.nodes]] = x
		self.update_attributes()

	def update_attributes(self):
		# extract upper triangle of adjacency matrix 
		flat = self.adj[np.triu_indices(self.adj.shape[0], k = 1)].flatten()
		unique, counts = np.unique(flat, return_counts=True)
		self.edges = dict(zip(unique, counts)).get(1.0, 0)
		self.nodes = self.adj.shape[0]
		self.degree = self.adj.sum(axis = 0)

	# converts node names to node indexs
	def node_index(self, names):
		return (self.node_names[names[0]], self.node_names[names[1]])

	# fills all of the adj matrix with 1s (except diagonal)
	def make_complete(self):
		self.adj.fill(1)
		np.fill_diagonal(self.adj, 0)
		self.update_attributes()


class UndirectedGraph(Graph):
	def __init__(self, size, names = None):
		super(UndirectedGraph, self).__init__(size, names = None)

	def add_edge(self, nodes):
		node_ind = self.node_index(nodes)
		if len(node_ind) == 2 and node_ind[0] >= 0 and node_ind[1] >= 0:
			self.adj[node_ind[0], node_ind[1]] = 1
			# maintain symmetry
			self.adj[node_ind[1], node_ind[0]] = 1
			self.update_attributes()
		else:
			print('Error in Node Name')

	def remove_edge(self, nodes):
		if len(self.node_index(nodes)) == 2:
			if self.adj[node_ind[0], node_ind[1]] == 1:
				self.adj[node_ind[0], node_ind[1]] = 0
				# maintain symmetry
				self.adj[node_ind[1], node_ind[0]] = 0
			else:
				print('No edge to remove!') 
			self.update_attributes()
		else:
			print('Please supply two nodes!')


class DirectedGraph(Graph):
	def __init__(self):
		super(DirectedGraph, self).__init__(size, names = None)

	def add_edge(self, nodes):
		node_ind = self.node_index(nodes)
		if len(node_ind) == 2 and node_ind[0] >= 0 and node_ind[1] >= 0:
			self.adj[node_ind[0], node_ind[1]] = 1
			self.update_attributes()
		else:
			print('Error in Node Name')

	def remove_edge(self, nodes):
		if len(self.node_index(nodes)) == 2:
			if self.adj[node_ind[0], node_ind[1]] == 1:
				self.adj[node_ind[0], node_ind[1]] = 0
			else:
				print('No edge to remove!') 
			self.update_attributes()
		else:
			print('Please supply two nodes!')


if __name__ == '__main__':
	g = gen.Generator.tree(5)
	print(tabulate(att.gather_stats(g)))
	vis.Visualise.draw_fruchterman_reingold(g)
