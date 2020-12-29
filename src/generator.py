import numpy as np
from hamilton import UndirectedGraph

class Generator:
	# symmetric binary tree
	def tree(depth):
		g = UndirectedGraph(0)
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
	def cycle(n):
		g = UndirectedGraph(n)
		for x in range(0, n - 1):
			g.add_edge((x, x + 1))
		#final edge to complete the cycle
		g.add_edge((n - 1, 0))
		return g

	# https://web.archive.org/web/20150824235818/
	# http://www3.nd.edu/~networks/Publication%20Categories/03%20Journal%20Articles/Physics/StatisticalMechanics_Rev%20of%20Modern%20Physics%2074,%2047%20(2002).pdf 
	# (p.71)
	def barabasi_albert(mo = 3, m = 2, lim = 50):
		g = UndirectedGraph(mo)
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
	def watts_strogatz(n, k = 4, b = 0.5):
		g = Generator.generate_cycle(n)
		# adding extra edges to the latice
		if k > 2:	
			for x in range(0, n):
				# no error will be thrown if we add mulitple edges, it will remain a single edge
				# number of edges to be added on each side of each node
				#  (minus 1 as we have already crated the cycle)
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
						# choosing which edge to add 
						e_add = np.random.choice(poss, 1, p = prob)[0]
						g.add_edge((x, e_add))
		return g

	def complete(n):
		g = UndirectedGraph(n)
		g.make_complete()
		return g 

	# G(n, p) form 
	def er_graph(n, p = 0.5):
		g = UndirectedGraph(n)
		# get the lower triangle of the adjacency matrix and
		# decide with a particular probability to add edges or not
		v = np.array([np.random.choice([1, 0], p = [p, 1 - p]) for xi in g.adj[np.triu_indices(n, k = 1)]])
		g.adj = np.zeros((n, n))
		g.adj[np.triu_indices(n, k = 1)] = v
		g.adj = g.adj + g.adj.T
		g.update_attributes()
		return g