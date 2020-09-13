import unittest
import networkx as nx
from hamilton import Graph

# this is the python class inheritance syntax 
class TestGraphs(unittest.TestCase):
    # the idea is to compare the results of my program
        # to the results produced by networkx
        # setup the tests
    def setUp(self):
        # self.tree = Generator.generate_tree(5)
        # self.cycle = Generator.generate_cycle(20)
        # self.complete = Generator.generate_complete(30)
            self.custom = Graph(4)
            self.custom.add_edge((0, 1))
            self.custom.add_edge((1, 2))
            self.custom.add_edge((2, 3))
            self.custom.add_edge((3, 1))
            self.custom.add_edge((3, 2))
            self.custom.add_edge((2, 1))

            # self.nx_tree = generators.classic.full_rary_tree(2, 5)
            # self.nx_cycle = generators.classic.cycle_graph(20)
            # self.nx_complete = generators.classic.complete_graph(30)
            self.nx_custom = nx.Graph()
            self.nx_custom.add_nodes_from([0, 1, 2, 3])
            self.nx_custom.add_edges_from([(0, 1), (1, 2), 
                                           (2, 3), (3, 1),
                                           (3, 2), (2, 1)])

    # first, lets just look at the custom graph 
    def test_nodes(self):
        self.assertEqual(self.custom.nodes, len(list(self.nx_custom.nodes)))

    def test_edges(self):
        self.assertEqual(self.custom.edges, len(list(self.nx_custom.edges)))

    def test_aand(self):
        self.assertEqual(self.custom.get_aand(), nx.average_neighbor_degree(self.nx_custom))

    def test_eig_cent(self):
        self.assetEqual(self.custom.eigenvector_centrality(), nx.eigenvector_centrality(self.nx_custom))

    def test_cycle(self):
        # does nx think a cycle exists 
        try:
            cycle = nx.algorithms.cycles.find_cycle(self.nx_custom)
            self.assertTrue(self.custom.spanning_trees() > 0)
        except nx.exception.NetworkXNoCycle:
            self.assertTrue(self.custom.spanning_trees() == 0)

    def test_connected(self):
        if nx.algorithms.components.is_connected(self.nx_custom):
            self.assertTrue(self.custom.is_connected(0))
        else: 
            self.assertFalse(self.custom.is_connected(0))

    def test_bipartite(self):
        if nx.algorithms.bipartite.is_bipartite(self.nx_custom):
            self.assertTrue(self.custom.is_bipartite())
        else:
            self.assertFalse(self.custom.is_bipartite())

    def test_regular(self):
        if nx.algorithms.regular.is_regular(self.nx_custom):
            self.assertTrue(self.custom.is_regular())
        else:
            self.assertFalse(self.custom.is_regular())

if __name__ == '__main__':
        unittest.main()

