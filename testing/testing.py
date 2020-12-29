import unittest
import networkx as nx
import math
import sys

sys.path.insert(0, '../src/')
import generator as gen
import hamilton as ham
import attributes as att

# this is the python class inheritance syntax 
class TestGraphs(unittest.TestCase):
    # the idea is to compare the results of my program
    # to the results produced by networkx
    # setup the tests
    def setUp(self):
        return 

graphs = {'tree': [gen.Generator.tree(5), nx.generators.classic.full_rary_tree(2, 31)],
          'cycle': [gen.Generator.cycle(20), nx.generators.classic.cycle_graph(20)],
          'complete': [gen.Generator.complete(30), nx.generators.classic.complete_graph(30)],
          'custom': [ ham.UndirectedGraph(4), nx.Graph()]
          }

graphs['custom'][0].add_edge((0, 1))
graphs['custom'][0].add_edge((1, 2))
graphs['custom'][0].add_edge((2, 3))
graphs['custom'][0].add_edge((3, 1))
graphs['custom'][0].add_edge((3, 2))
graphs['custom'][0].add_edge((2, 1))

graphs['custom'][1].add_nodes_from([0, 1, 2, 3])
graphs['custom'][1].add_edges_from([(0, 1), (1, 2), 
                                    (2, 3), (3, 1),
                                    (3, 2), (2, 1)])

# h is the hamilton graph, n in the network graph 
def create_nodes(h, n):
    def nodes(self):
        self.assertEqual(h.nodes, len(list(n.nodes)))
    return nodes

def create_edges(h, n):
    def edges(self):
        self.assertEqual(h.edges, len(n.edges.data()))
    return edges

def create_aand(h, n):
    def aand(self):
        self.assertEqual(att.aand(h), nx.average_neighbor_degree(n))
    return aand

def create_eig(h, n):
    def eig_cent(self):
        heig = [round(x, 3) for x in att.eigenvector_centrality(h).tolist()]
        neig = [round(x, 3) for x in list(nx.eigenvector_centrality(n).values())]
        self.assertCountEqual(heig, neig)
    return eig_cent

def create_cycle(h, n):
    def cycle(self):
        # does nx think a cycle exists 
        try:
            cycle = nx.algorithms.cycles.find_cycle(n)
            self.assertTrue(att.spanning_trees(h) > 0)
        except nx.exception.NetworkXNoCycle:
            self.assertTrue(att.spanning_trees(h) == 0)
    return cycle

def create_connected(h, n):
    def connected(self):
        if nx.algorithms.components.is_connected(n):
            self.assertTrue(att.is_connected(h))
        else: 
            self.assertFalse(att.is_connected(h))
    return connected

def create_bipartite(h, n):
    def bipartite(self):
        if nx.algorithms.bipartite.is_bipartite(n):
            self.assertTrue(att.is_bipartite(h))
        else:
            self.assertFalse(att.is_bipartite(h))
    return bipartite

def create_regular(h, n):
    def regular(self):
        if nx.algorithms.regular.is_regular(n):
            self.assertTrue(att.is_regular(h))
        else:
            self.assertFalse(att.is_regular(h))
    return regular

for i, (k, v) in enumerate(graphs.items()):
    setattr(TestGraphs, 'test_expected_%s_node' % k, create_nodes(v[0], v[1])) 
    setattr(TestGraphs, 'test_expected_%s_edges' % k, create_edges(v[0], v[1]))
    setattr(TestGraphs,  'test_expected_%s_aand' % k, create_aand(v[0], v[1]))
    setattr(TestGraphs, 'test_expected_%s_eig' % k, create_eig(v[0], v[1])) 
    setattr(TestGraphs, 'test_expected_%s_cycle' % k, create_cycle(v[0], v[1])) 
    setattr(TestGraphs, 'test_expected_%s_connected' % k, create_connected(v[0], v[1])) 
    setattr(TestGraphs, 'test_expected_%s_bipartite' % k, create_bipartite(v[0], v[1])) 
    setattr(TestGraphs, 'test_expected_%s_regular' % k, create_regular(v[0], v[1])) 



