import numpy as np

def gather_stats(g):
	stat = []
	stat.append(['Nodes', g.nodes])
	stat.append(['Edges', g.edges])
	stat.append(['Eigenvector Centrality of node 1', eigenvector_centrality(g)[0]])
	stat.append(['Has cycle', has_cycle(g)])
	stat.append(['Spanning trees', spanning_trees(g)])
	stat.append(['aand of node 1', aand(g)[0]])
	stat.append(['probability of node w/ degree 0', degree_dist(g, 0)])
	stat.append(['Global clustering coeff is ', global_clustering_coeff(g)])
	stat.append(['Is bipartite: ', is_bipartite(g)])
	stat.append(['Is regular : ', is_regular(g)])
	stat.append(['Degree of node 1', g.degree[0]])
	stat.append(['Connected ', is_connected(g)])
	return stat

def get_laplace(graph):
	#first calculate the degree matrix 
	D = np.diag(graph.degree)
	#find laplacian matrix
	return D - graph.adj		
		
#dfs helper 
def dfs_bipartite_helper(graph, v, discovered, colour):
	# for every edge v -> u 
	# vals of nodes which are adjacent to v
	ind = np.where(graph.adj[v] == 1)[0]
	for x in ind:
		if x not in discovered:
			discovered.append(x)
			colour[x] = not colour[v]
			if not dfs_bipartite_helper(graph, x, discovered, colour):
				return False
		elif colour[x] == colour[v]:
			return False
	return True 

# returns an array giving the average nearest neighbour degree for each node in a graph
def aand(graph):
	aand = {}
	for x in range(0, graph.nodes):
		aand[x] = graph.degree[np.where(graph.adj[x] == 1)[0]].mean()
	return aand

def eigenvector_centrality(graph):
	val, vec = np.linalg.eig(graph.adj)
	maxvec = vec[:, np.argmax(val)]
	real = np.isreal(maxvec)
	return np.abs(maxvec[real])

# https://stackoverflow.com/questions/16436165/detecting-cycles-in-an-adjacency-matrix
def has_cycle(graph):
	if graph.nodes < 3:
		return False
	#first calculate the degree matrix 
	D = np.diag(graph.adj.sum(axis = 0))
	#find laplacian matrix
	L = D - graph.adj
	if 0.5 * np.trace(L) == np.linalg.matrix_rank(L):
		return False
	else:
		return True

# using depth first search to determine if a graph is bipartite
# if this returns false an odd cycle is implied
# come back to this, I need to understand the algo better before trying to implement
def is_bipartite(graph, root = 0):
	if graph.edges == 0:
		return True
	discovered = [0]
	colour = {0: True}
	return dfs_bipartite_helper(graph, root, discovered, colour)

# conduct a depth first search,
def is_connected(graph, root = 0):
	s = []
	visited = []
	s.append(root)
	visited.append(root)
	while(len(s) > 0):
		v = s.pop()
		it = np.nditer(graph.adj[:,v], flags=['f_index'])
		for x in it:
			if x == 1 and it.index not in visited:
					s.append(it.index)                     
					visited.append(it.index)
	return len(visited) == graph.nodes 

# this method only works for undirected graphs
# https://math.stackexchange.com/questions/2133814/calculation-transitivity-of-a-graph-from-the-adj-matrix
# https://en.wikipedia.org/wiki/Clustering_coefficient#cite_note-4
# binary global clustering coefficient 
def global_clustering_coeff(graph):
	# calculate number of closed triplets (or 3 times triangles)
	cldtrip = (np.trace(np.linalg.matrix_power(graph.adj, 3))) / 2
	#calculate total number of triplets, total number of triplets (both open and closed).
	triples = ((np.linalg.matrix_power(graph.adj, 2)).sum()
				- np.trace(np.linalg.matrix_power(graph.adj, 2))) / 2 
	return cldtrip / triples

def is_regular(graph):
	return len(set(graph.degree)) == 1

# function which determines if a graph is complete
# def is_complete

#spanning tree visits every vertice using the minimum number of edges
#uses kirchoffs thereom to calculate the number of spanning trees in a graph
#number of spanning trees equal to any cofactor of the laplacian
def spanning_trees(graph):
	L = get_laplace(graph)
	Q = L[1:, 1:]
	det = np.linalg.det(Q)
	return det

def degree_dist(graph, k):
	return (graph.degree == k).sum() / graph.degree.shape[0]