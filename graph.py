import numpy as np

class Generator:
	#generates a symmetric binary tree
	def generate_tree(depth):
		g = Graph(0)
		for x in range(0, depth):
			# n is new nodes being added to this layer
			n = 2 ** x
			print('adding nodes : ', n)
			g.add_nodes(n)
			#skip for the root
			if n != 1:
				#j counts the number of connections to parent
				j = 0
				# inital parent is the first node in the row above
				parent = int(g.nodes - n - (n / 2))
				for i in range(g.nodes - n, g.nodes):
					if j == 2:
						#reset j
						j = 0 
						#move to next parent
						parent += 1
					g.add_edge((parent, i))
					j += 1
		return g

	#generates a cycle graph
	def generate_cycle(n):
		g = Graph(n)
		for x in range(0, n - 1):
			g.add_edge((x, x + 1))
		#final edge to complete the cycle
		g.add_edge((n - 1, 0))
		return g

	def generate_complete(n):
		g = Graph(n)
		g.make_complete()
		return g 

	# generating the G(n, p) form of a er graph
	# use np.random.choice to decide the probability 
	def generate_erdos_renyi(n, p):
		g = Graph(n)


# assuming undirected, for now 
# assuming unweighted for now 
# node label start at zero, they are the same indexes 
class Graph:
	def __init__(self, size):	
		self.adj = np.zeros((size or 0, size or 0))
		num_rows = self.adj.shape[0] 
		self.nodes = num_rows
		self.edges = 0 
		self.update_attributes()

	def add_nodes(self, count):
		# will add a new row & column of zeroes to the adjacency matrix 
		num_rows = self.adj.shape[0] 
		b = np.zeros((num_rows + count, num_rows + count))
		b[:num_rows, :num_rows] = self.adj
		self.adj = b
		self.nodes += count 
		self.update_attributes()

	def add_edge(self, nodes):
		if len(nodes) == 2 and nodes[0] >= 0 and nodes[1] >= 0:
			self.adj[nodes[0], nodes[1]] = 1
			#to maintain symmettry
			self.adj[nodes[1], nodes[0]] = 1
			self.edges += 1 
			self.update_attributes()

	# fills all of the adj matrix with 1s (except diagonal)
	def make_complete(self):
		self.adj.fill(1)
		np.fill_diagonal(self.adj, 0)
		self.edges = int((self.nodes * (self.nodes - 1)) / 2)
		self.update_attributes()

	#https://en.wikipedia.org/wiki/Eigenvector_centrality
	#measure of the influence of node in a network 
	#come back to this later - I'm not sure I understand enough about linalg, or maybe I need to find a detailed textbook explanation
	def eigenvector_centrality(self):
		eigenvalue, eigenvector = np.linalg.eig(self.adj)
		np.set_printoptions(precision=3)
		largest = eigenvector.real
		norm = np.sign(largest.sum()) * np.linalg.norm(largest)
		return (largest / norm)[:,np.argmax(eigenvalue)] 

	#from https://stackoverflow.com/questions/16436165/detecting-cycles-in-an-adjacency-matrix
	def has_cycle(self):
		if self.nodes < 3:
			return False
		#first calculate the degree matrix 
		D = np.diag(self.adj.sum(axis = 0))
		#find laplacian matrix
		L = D - self.adj
		if 0.5 * np.trace(L) == np.linalg.matrix_rank(L):
			return False
		else:
			return True

	#using depth first search to determine if a graph is bipartite
	#if this returns false an odd cycle is implied
	#come back to this, I need to understand the algo better before trying to implement
	def is_bipartite(self, root):
		if self.edges == 0:
			return True
		s = []
		flag = True 
		visited = []
		colouring = {}
		s.append(root)
		visited.append(root)
		print('visited ', root)
		colouring[root] = 0
		# root is zero and parent is -1 (doesnt exist for now)
		v = None
		while(len(s) > 0):
			if v is None: 
				p = -1
			else:
				p = v
				v = s.pop()  
				colouring[v] = 1 - colouring[p]			
			it = np.nditer(self.adj[:,v], flags=['f_index'])
			for x in it:
				if x == 1 and it.index not in visited:
					s.append(it.index)
					visited.append(it.index)
					print('visited ', it.index)
		return colouring


	# conduct a Depth first search, if the number of nodes found equals the number of nodes in the graph, the graph is connected
	def is_connected(self, root):
		s = []
		visited = []
		s.append(root)
		visited.append(root)
		while(len(s) > 0):
			v = s.pop()
			it = np.nditer(self.adj[:,v], flags=['f_index'])
			for x in it:
				if x == 1 and it.index not in visited:
						s.append(it.index)                     
						visited.append(it.index)
		return len(visited) == self.nodes 

	def is_regular(self):
		return len(set(self.degree)) == 1

	# function which determines if a graph is complete
	# def is_complete

	#spanning tree visits every vertice using the minimum number of edges
	#uses kirchoffs thereom to calculate the number of spanning trees in a graph
	#number of spanning trees equal to any cofactor of the laplacian
	def spanning_trees(self):
		#first calculate the degree matrix 
		D = np.diag(self.degree)
		#find laplacian matrix
		L = D - self.adj
		Q = L[1:, 1:]
		det = np.linalg.det(Q)
		return det

	#finds the probability of a random node having a particular degree. 
	def degree_dist(self, k):
		return (self.degree == k).sum() / self.degree.shape[0]

	def update_attributes(self):
		self.degree = self.adj.sum(axis = 0)


g = Generator.generate_tree(10)
print(g.eigenvector_centrality())
print('has cycle : ', g.has_cycle())
print('spanning trees : ', g.spanning_trees())
print('probability of randomly selecting a node with degree = 0 is ', g.degree_dist(0))
print('NODES', g.nodes)
print('EDGES', g.edges)
print(g.adj)
print('this graph is regular : ', g.is_regular())
print('the degrees of nodes are ', g.degree)
print('connected ', g.is_connected(0))
