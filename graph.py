import numpy as np
import math
from graphics import *

class Visualise:
	c1 = 2
	c2 = 1
	c3 = 1000
	c4 = 0.1

	# calculate the force on a vertex 
	# use a logarithmic rather than linear scale
	# returns the force
	# where d is the distance between the two vertices
	# it may be better to calculate d within the calculate method, let us wait and see
	def spring_force(d):
		if d == 0:
			return 0
		if d < Visualise.c2:
			return 0
		return Visualise.c1 * math.log(d / Visualise.c2, 10)

	# use an inverse square law for the inverse force
	# returns the force  
	def repel_force(d):
		if d == 0:
			return 0
		return Visualise.c3 / (d**2)

	# draw a graph using the spectral layout
	# https://en.wikipedia.org/wiki/Spectral_layout
	def draw_spectral(g):
		np.set_printoptions(precision=3)
		L = g.get_laplace()
		eigenvalue, eigenvector = np.linalg.eig(L)
		# returns the indices of the two smallest eigenvalues
		minind = np.argsort(eigenvalue)[1 : 3]
		xeig = eigenvector[:,minind[0]]
		yeig = eigenvector[:,minind[1]]
		cords = {}
		# record max cords val to set the cord max in graphics
		maxc = max(max(xeig), max(yeig))
		minc = min(min(xeig), min(yeig))
		# create dict of x, y tuples to hold co-ordinates for each vertice
		for x in range(0, g.nodes):
			cords[x] = (xeig[x], yeig[x])
		win = GraphWin(width = 800, height = 800) # create a window
		buffer = 1.05
		win.setCoords(minc * buffer, minc * buffer, maxc * buffer, maxc * buffer) # set the coordinates of the window; bottom left is (0, 0) and top right is (10, 10)
		# then add edges
		ui =  np.triu_indices(g.nodes, k = 1)
		for e in range(0, len(ui[0])):
			if g.adj[ui[0][e], ui[1][e]] == 1:
				l = Line(Point(cords[ui[0][e]][0], cords[ui[0][e]][1]), Point(cords[ui[1][e]][0], cords[ui[1][e]][1]))
				l.draw(win)
		# then, nodes
		for i in cords:
			c = Circle(Point(cords[i][0], cords[i][1]), maxc / 20)	
			message = Text(Point(cords[i][0], cords[i][1]), i)
			message.draw(win)
			c.draw(win) 
		win.getMouse() # pause before closing


	def fruchterman_reingold(A, fixed=None, iterations=50, threshold=1e-4, dim=2):

		nnodes, _ = A.shape

		# random initial positions
		pos = np.asarray(np.random.rand(nnodes, dim), dtype=A.dtype)

		# optimal distance between nodes
		k = np.sqrt(1.0 / nnodes)
		# the initial "temperature"  is about .1 of domain area (=1x1)
		# this is the largest step allowed in the dynamics.
		# We need to calculate this in case our fixed positions force our domain
		# to be much bigger than 1x1
		t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
		# simple cooling scheme.
		# linearly step down by dt on each iteration so last iteration is size dt.
		dt = t / float(iterations + 1)
		delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
		# the inscrutable (but fast) version
		# this is still O(V^2)
		# could use multilevel methods to speed this up significantly
		for iteration in range(iterations):
			# matrix of difference between points
			delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
			# distance between points
			distance = np.linalg.norm(delta, axis=-1)
			# enforce minimum distance of 0.01
			np.clip(distance, 0.01, None, out=distance)
			# displacement "force"
			displacement = np.einsum(
				"ijk,ij->ik", delta, (k * k / distance ** 2 - A * distance / k)
			)
			# update positions
			length = np.linalg.norm(displacement, axis=-1)
			length = np.where(length < 0.01, 0.1, length)
			delta_pos = np.einsum("ij,i->ij", displacement, t / length)
			if fixed is not None:
				# don't change positions of fixed nodes
				delta_pos[fixed] = 0.0
			pos += delta_pos
			# cool temperature
			t -= dt
			err = np.linalg.norm(delta_pos) / nnodes
			if err < threshold:
				break


		win = GraphWin(width = 800, height = 800) # create a window
		# coords on x and y axis are between 0 and 100 
		win.setCoords(np.amin(pos) * 1.1, np.amin(pos) * 1.1, np.amax(pos) * 1.1, np.amax(pos) * 1.1) 

		# # then render edges 
		ui = np.triu_indices(nnodes, k = 1)
		for e in range(0, len(ui[0])):
			if g.adj[ui[0][e], ui[1][e]] == 1:
				l = Line(Point(pos[ui[0][e],0], pos[ui[0][e], 1]), Point(pos[ui[1][e], 0], pos[ui[1][e], 1]))
				l.draw(win)
		# render nodes
		for i in range(0, nnodes):
			c = Circle(Point(pos[i, 0], pos[i, 1]), 0.05)	
			c.draw(win) 
		win.getMouse() # pause before closing


		return pos


	# # # draw a graph on graphics.py with the force directed layout
	# # # these can also be known as spring embedders
	# # # hookes law like forces between adjacent nodes, except on a logarithmic rather than linear scale 
	# # # we make non-adjacent vertices repel eachother with an inverse square law
	# # # http://cs.brown.edu/people/rtamassi/gdhandbook/chapters/force-directed.pdf
	# def draw_force(g):	
	# 	# create dict of x, y tuples to hold co-ordinates for each vertice
	# 	cords = {}
	# 	# first give all nodes random locations in our coord space 
	# 	for x in range(0, g.nodes):
	# 		rng = np.random.default_rng()
	# 		cords[x] = (rng.integers(0, high = 100, endpoint = True, size = 1)[0], rng.integers(0, high =  100, endpoint = True, size = 1)[0])
	# 	# run the simulation M times, this is usually enough to reach a minimum energy state
	# 	M = 100

	# 	for s in range(0, M):
	# 		print('cords ', cords)
	# 		# #simulate the system, iterate over all nodes 
	# 		for i in range(0, g.nodes):
	# 			# for each vertice, first find the combined repellant force of all non adjacent vertices on each node
	# 			# vals of nodes which are adjacent to i
	# 			na = np.array(range(0, g.nodes))
	# 			index = np.argwhere(na==i)
	# 			naind = np.delete(na, index)
	# 			print('naind for ', i,  naind)
	# 			cum_repellent_force = np.array([0, 0])
	# 			# loop over all non-adjacent nodes 
	# 			for x in naind:
	# 				delta_x = cords[x][0] - cords[i][0]
	# 				delta_y = cords[x][1] - cords[i][1]

	# 				# print('delta x', delta_x)
	# 				# print('delta y', delta_y)
	# 				# first calculate the current euclidean distance between the two nodes
	# 				d = abs((delta_x**2 + delta_y**2) ** (1/2))
	# 		# 		# get the magnitude of the repel force 
	# 				r = Visualise.repel_force(d)
	# 		# 		# get the direction of the repel force as an angle where 0 radians = 3 oclock
	# 				theta_radians = (math.atan2(delta_y, delta_x) - math.pi) % (2 * math.pi) 
	# 		# 		# find the component vectors 
	# 				r_x, r_y = -1 * r * np.cos(theta_radians), -1 * r * np.sin(theta_radians)
	# 		# 		# print('rx ry', r_x, r_y)
	# 		# 		# a baby crawls before it walks 
	# 				cum_repellent_force = np.add(cum_repellent_force, np.array([r_x, r_y]))

	# 			# secondly, find the cumulative attractive force with all adjacent 
	# 			aind = np.where(g.adj[i] == 1)[0]
	# 			cum_attr_force = np.array([0, 0])
	# 			# loop over all adjacent nodes 
	# 			#print(' adj indices for ', i, aind)
	# 			for j in aind:
	# 				delta_x = cords[j][0] - cords[i][0]
	# 				delta_y = cords[j][1] - cords[i][1]

	# 				# first calculate the current euclidean distance between the two nodes
	# 				d = abs((delta_x**2 + delta_y**2) ** (1/2))
	# 				# get the magnitude of the spring force 
	# 				r = Visualise.spring_force(d)
	# 				# get the direction of the spring force as an angle where 0 radians = 3 oclock
	# 				theta_radians = math.atan2(delta_y, delta_x)
	# 				#print('radians', theta_radians)
	# 				# find the component vectors 
	# 				s_x, s_y = r * np.cos(theta_radians), r * np.sin(theta_radians)
	# 				#print('components', s_x, s_y)
	# 				# a baby crawls before it walks 
	# 				cum_attr_force = np.add(cum_attr_force, np.array([s_x, s_y]))
	# 			# then, find the net force of the repellant and attractive forces and move the node by this ammout
	# 			net_force = np.add(cum_repellent_force, cum_attr_force)
	# 			print('cum repellant force ', cum_repellent_force)
	# 			print('cum spring force ', cum_attr_force)
	# 			print('net force', net_force)
	# 			# update the coord based on the force
	# 			cords[x] = (cords[x][0] + Visualise.c4*net_force[0], cords[x][1] + Visualise.c4*net_force[1])

	# 	win = GraphWin(width = 800, height = 800) # create a window
	# 	# coords on x and y axis are between 0 and 100 
	# 	win.setCoords(-150, -150, 150, 150) 

	# 	# then render edges 
	# 	ui =  np.triu_indices(g.nodes, k = 1)
	# 	for e in range(0, len(ui[0])):
	# 		if g.adj[ui[0][e], ui[1][e]] == 1:
	# 			l = Line(Point(cords[ui[0][e]][0], cords[ui[0][e]][1]), Point(cords[ui[1][e]][0], cords[ui[1][e]][1]))
	# 			l.draw(win)
	# 	# render nodes
	# 	for i in cords:
	# 		c = Circle(Point(cords[i][0], cords[i][1]), 40)	
	# 		message = Text(Point(cords[i][0], cords[i][1]), i)
	# 		message.draw(win)
	# 		c.draw(win) 
	# 	win.getMouse() # pause before closing



class Generator:
	#generates a symmetric binary tree
	def generate_tree(depth):
		g = Graph(0)
		for x in range(0, depth):
			# n is new nodes being added to this layer
			n = 2 ** x
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

	# start with 3 nodes and two edges
	# https://web.archive.org/web/20150824235818/
	# http://www3.nd.edu/~networks/Publication%20Categories/03%20Journal%20Articles/Physics/StatisticalMechanics_Rev%20of%20Modern%20Physics%2074,%2047%20(2002).pdf 
	# (p.71)
	def generate_barabase_albert(mo = 3, m = 2, lim = 50):
		g = Graph(mo)
		g.add_edge((0,1))
		g.add_edge((1,2))

		while(g.nodes < lim):
			g.add_nodes(1)
			# possible partner nodes  
			# g is minus 1 below to avoid choosing the node we just added to the graph 
			partners = range(0, g.nodes - 1)
			#probability of connecting to each of these partners
			prob = []
			for x in partners:
				prob.append(g.degree[x] / (g.edges * 2))
			# add m edges to separate existing nodes
			edg = np.random.choice(partners, m, replace = False, p = prob)
			for i in edg:
				g.add_edge((g.nodes - 1, i))
		return g

	#https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model
	# the mean degree k is assumed to be an even integer
	# b must be between 0 and 1 inclusive 
	# need to verify that this method is correct 
	def generate_watts_strogatz(n, k = 4, b = 1):
		g = Generator.generate_cycle(n)
		# adding extra edges to the latice
		if k > 2:	
			for x in range(0, n):
				# no error will be thrown if we add mulitple edges, it will remain a single edge
				# number of edges to be added on each side of each node (minus 1 as we have already crated the cycle)
				edg = int(k / 2) - 1
				for i in range(0, edg):
					# determine which other node to add the edge to
					# specifically, what is the index step between the two nodes
					step = 2 + i 
					if x + step > g.nodes - 1:
						# things get messier when adding edges at the end of the cycle
						g.add_edge((x, x + step - g.nodes))
					else:
						g.add_edge((x, x + step))
		# now, the rewiring stage
		# iterate over each node and each right edge 
		for x in range(0, n):
			for i in range(1, int(k / 2) + 1):
				if np.random.choice([1, 0], 1, p = [b, 1 - b])[0] == 1:
					if x + i > g.nodes - 1:
						#remove edge being rewired 
						g.remove_edge((x, x + i - g.nodes))
						nodes = list(range(0, g.nodes))
						nodes.remove(x)
						icd = set(np.where(g.adj[x] == 1)[0])
						poss = list(filter(lambda x: x not in icd, nodes))
						# probability of choosing any of the k new edges
						prob = [1 / len(poss)] * len(poss)
						g.add_edge((x, np.random.choice(poss, 1, p = prob)[0]))						
					else:	
						#remove edge being rewired 
						g.remove_edge((x, x + i))
						nodes = list(range(0, g.nodes))
						nodes.remove(x)
						icd = set(np.where(g.adj[x] == 1)[0])
						poss = list(filter(lambda x: x not in icd, nodes))
						# probability of choosing any of the k new edges
						prob = [1 / len(poss)] * len(poss)
						g.add_edge((x, np.random.choice(poss, 1, p = prob)))
		return g


	def generate_complete(n):
		g = Graph(n)
		g.make_complete()
		return g 

	# generating the G(n, p) form of a er graph
	# use np.random.choice to decide the probability 
	def generate_er_graph(n, p = 0.5):
		g = Graph(n)
		# get the lower triangle of the adjacency matrix and
		# decide with a particular probability to add edges or not
		v = np.array([np.random.choice([1, 0], p = [p, 1 - p]) for xi in g.adj[np.triu_indices(n, k = 1)]])
		g.adj = np.zeros((n, n))
		g.adj[np.triu_indices(n, k = 1)] = v
		g.adj = g.adj + g.adj.T
		g.update_attributes()
		return g

# node label start at zero, they are the same indexes 
class Graph:
	def __init__(self, size, names = None):	
		# node names stored as a dictionary, relating a node name #dict key to a node index # dict value
		self.node_names = {}
		if names is None:
			self.custom_node_names = False
			for x in range(0, size):
				self.node_names[x] = x
		else:
			self.custom_node_names = True
			if len(names) == size:
				for x in range(0, size):
					self.node_names[names[x]] = x
		#include support for other kinds of node naming here
		self.adj = np.zeros((size or 0, size or 0))
		num_rows = self.adj.shape[0] 
		self.update_attributes()

	def get_laplace(self):
		#first calculate the degree matrix 
		D = np.diag(self.degree)
		#find laplacian matrix
		return D - self.adj		
			
	#dfs helper 
	def dfs_bipartite_helper(self, v, discovered, colour):
		# for every edge v -> u 
		# vals of nodes which are adjacent to v
		ind = np.where(self.adj[v] == 1)[0]
		for x in ind:
			if x not in discovered:
				discovered.append(x)
				colour[x] = not colour[v]
				if not self.dfs_bipartite_helper(x, discovered, colour):
					return False
			elif colour[x] == colour[v]:
				return False
		return True 

	# returns an array giving the average nearest neighbour degree for each node in a graph
	# is the neighbourhood open or closed? I will assume open
	def get_aand(self):
		aand = {}
		for x in range(0, self.nodes):
			aand[x] = self.degree[np.where(self.adj[x] == 1)[0]].mean()
		return aand

	# add nodes to the current graph
	def add_nodes(self, size, names = None):
		if names is None:
			# standard case, no custom node names defined 
			# will add a new row & column of zeroes to the adjacency matrix 
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

	def add_edge(self, nodes):
		# convert node name to a node index 
		node_ind = (self.node_names[nodes[0]], self.node_names[nodes[1]])
		if len(node_ind) == 2 and node_ind[0] >= 0 and node_ind[1] >= 0:
			self.adj[node_ind[0], node_ind[1]] = 1
			#to maintain symmetry
			self.adj[node_ind[1], node_ind[0]] = 1
			self.update_attributes()
		else:
			print('Some function parameter error')

	def remove_edge(self, nodes):
		# convert node name to a node index 
		node_ind = (self.node_names[nodes[0]], self.node_names[nodes[1]])
		if len(node_ind) == 2:
			if self.adj[node_ind[0], node_ind[1]] == 1:
				self.adj[node_ind[0], node_ind[1]] = 0
				#to maintain symmetry
				self.adj[node_ind[1], node_ind[0]] = 0
			else:
				print('No edge to remove!') 
			self.update_attributes()
		else:
			print('Please supply two nodes!')

	# fills all of the adj matrix with 1s (except diagonal)
	def make_complete(self):
		self.adj.fill(1)
		np.fill_diagonal(self.adj, 0)
		self.update_attributes()

	# https://en.wikipedia.org/wiki/Eigenvector_centrality
	# measure of the influence of node in a network 
	# come back to this later
	# I'm not sure I understand enough about linalg, or maybe I need to find a detailed textbook explanation
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

	# using depth first search to determine if a graph is bipartite
	# if this returns false an odd cycle is implied
	# come back to this, I need to understand the algo better before trying to implement
	def is_bipartite(self, root = 0):
		if self.edges == 0:
			return True
		discovered = [0]
		colour = {0: True}
		return self.dfs_bipartite_helper(root, discovered, colour)

	# conduct a Depth first search,
	#  if the number of nodes found equals the number of nodes in the graph, the graph is connected
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

	# this method only works for undirected graphs
	# https://math.stackexchange.com/questions/2133814/calculation-transitivity-of-a-graph-from-the-adj-matrix
	# https://en.wikipedia.org/wiki/Clustering_coefficient#cite_note-4
	# binary global clustering coefficient 
	def global_clustering_coeff(self):
		# calculate number of closed triplets (or 3 times triangles)
		cldtrip = (np.trace(np.linalg.matrix_power(self.adj, 3))) / 2
		#calculate total number of triplets, total number of triplets (both open and closed).
		triples = ((np.linalg.matrix_power(self.adj, 2)).sum() - np.trace(np.linalg.matrix_power(self.adj, 2))) / 2 
		return cldtrip / triples

	def is_regular(self):
		return len(set(self.degree)) == 1

	# function which determines if a graph is complete
	# def is_complete

	#spanning tree visits every vertice using the minimum number of edges
	#uses kirchoffs thereom to calculate the number of spanning trees in a graph
	#number of spanning trees equal to any cofactor of the laplacian
	def spanning_trees(self):
		L = self.get_laplace()
		Q = L[1:, 1:]
		det = np.linalg.det(Q)
		return det

	#finds the probability of a random node having a particular degree. 
	def degree_dist(self, k):
		return (self.degree == k).sum() / self.degree.shape[0]

	# only works for graphs which are symmetric with an all zero diag (no looping)
	def update_attributes(self):
		# extract upper triangle of adjacency matrix 
		flat = self.adj[np.triu_indices(self.adj.shape[0], k = 1)].flatten()
		unique, counts = np.unique(flat, return_counts=True)
		self.edges = dict(zip(unique, counts)).get(1.0, 0)
		self.nodes = self.adj.shape[0]
		self.degree = self.adj.sum(axis = 0)

g = Generator.generate_tree(5)
print('egen centrality', g.eigenvector_centrality())
print('has cycle : ', g.has_cycle())
print('spanning trees : ', g.spanning_trees())
print('aand : ', g.get_aand())
print('probability of randomly selecting a node with degree = 0 is ', g.degree_dist(0))
print('NODES', g.nodes)
print('EDGES', g.edges)
print('The global clustering coeff is ', g.global_clustering_coeff())
print('The graph is bipartite: ', g.is_bipartite())
print('this graph is regular : ', g.is_regular())
print('the degrees of nodes are ', g.degree)
print('connected ', g.is_connected(0))
print('naming dict,', g.node_names)

print('fruchterman_reingold is', Visualise.fruchterman_reingold(g.adj))
