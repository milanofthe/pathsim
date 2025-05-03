########################################################################################
##
##                                   TESTS FOR 
##                                'utils/graph.py'
##
##                               Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.utils.graph import Graph
from pathsim.blocks import Block, Integrator, Amplifier, Adder
from pathsim.connection import Connection


# TESTS ================================================================================

class TestGraph(unittest.TestCase):
    """tests for the Graph class in pathsim.utils.graph"""

    def setUp(self):
        """Set up common resources for test methods."""
        
        # Algebraic Blocks (assume len=1)
        self.amp1 = Amplifier(gain=2) # len=1
        self.amp2 = Amplifier(gain=-1) # len=1
        self.add1 = Adder() # len=1
        self.add2 = Adder("+-") # Subtractor, len=1

        # Dynamic Blocks (assume len=0)
        self.int1 = Integrator(initial_value=0.0) # len=0
        self.int2 = Integrator(initial_value=1.0) # len=0

        # Longest algebraic path: amp1(1) -> add1(1) -> amp2(1) = length 3, ends at depth 3
        # Graph depth = max_node_depth + 1 = 3 + 1 = 4
        self.c_amp1_add1 = Connection(self.amp1, self.add1[0])
        self.c_int1_add1 = Connection(self.int1, self.add1[1])
        self.c_add1_amp2 = Connection(self.add1, self.amp2)
        self.c_amp2_int2 = Connection(self.amp2, self.int2)

        self.nodes_acyclic = [self.amp1, self.add1, self.int1, self.amp2, self.int2]
        self.conns_acyclic = [self.c_amp1_add1, self.c_int1_add1, self.c_add1_amp2, self.c_amp2_int2]

        # Loop: add2 -> amp1 -> add2
        self.c_add2_amp1 = Connection(self.add2, self.amp1)
        self.c_amp1_add2 = Connection(self.amp1, self.add2[1])

        self.nodes_alg_loop = [self.add2, self.amp1]
        self.conns_alg_loop = [self.c_add2_amp1, self.c_amp1_add2]

        # Loop: int1 -> int2 -> int1
        self.c_int1_int2 = Connection(self.int1, self.int2)
        self.c_int2_int1 = Connection(self.int2, self.int1)

        self.nodes_dyn_loop = [self.int1, self.int2]
        self.conns_dyn_loop = [self.c_int1_int2, self.c_int2_int1]


    def test_init_empty(self):
        """Test initializing an empty graph."""

        g = Graph([], [])
        self.assertEqual(len(g), 0)
        self.assertFalse(g.has_loops)

        alg_depth, loop_depth = g.depth()
        self.assertEqual(alg_depth, 0)
        self.assertEqual(loop_depth, 0)


    def test_init_with_mixed_nodes_edges_acyclic(self):
        """Test initializing a graph with mixed nodes and acyclic connections."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)
        self.assertEqual(len(g), 5)
        self.assertFalse(g.has_loops)


    def test_init_with_dynamic_loop(self):
        """Test initializing a graph with a dynamic loop."""

        g = Graph(self.nodes_dyn_loop, self.conns_dyn_loop)
        self.assertEqual(len(g), 2)
        self.assertFalse(g.has_loops)


    def test_init_with_algebraic_loop(self):
        """Test initializing a graph with an algebraic loop."""
        g = Graph(self.nodes_alg_loop, self.conns_alg_loop)
        self.assertEqual(len(g), 2)
        self.assertTrue(g.has_loops)


    def test_has_loops_mixed_false(self):
        """Test loop detection in an acyclic graph with mixed blocks."""
        g = Graph(self.nodes_acyclic, self.conns_acyclic)
        self.assertFalse(g.has_loops)


    def test_has_loops_mixed_true_algebraic(self):
        """Test loop detection with an algebraic loop present."""

        # Add unconnected int1 to ensure it doesn't interfere
        nodes = self.nodes_alg_loop + [self.int1]
        g = Graph(nodes, self.conns_alg_loop)
        self.assertTrue(g.has_loops)


    def test_has_loops_mixed_true_via_integrator(self):
        """Test loop detection when loop path includes integrator (should be false)."""

        # Loop: amp1 -> add1 -> amp2 -> int2 -> amp1
        # Connection needed: Connection(self.amp1, self.add1[0]) -> already have c_amp1_add1
        # Connection needed: Connection(self.add1, self.amp2) -> already have c_add1_amp2
        # Connection needed: Connection(self.amp2, self.int2) -> already have c_amp2_int2
        c_int2_amp1_new = Connection(self.int2, self.amp1)
        nodes = [self.amp1, self.add1, self.amp2, self.int2]
        connections = [self.c_amp1_add1, self.c_add1_amp2, self.c_amp2_int2, c_int2_amp1_new]
        g = Graph(nodes, connections)

        # Algebraic loop detection should return False because int2 breaks the algebraic path
        self.assertFalse(g.has_loops)


    def test_depth_acyclic_mixed(self):
        """Test depth calculation in an acyclic graph with mixed blocks."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)
        alg_depth, loop_depth = g.depth()

        # Max upstream algebraic path ends at amp2 or int2, tracing back through amp2(1)+add1(1)+amp1(1) = 3.
        # Graph._alg_depth is max_depth + 1.
        self.assertEqual(alg_depth, 4)
        self.assertEqual(loop_depth, 0)


    def test_depth_algebraic_loop(self):
        """Test depth calculation with an algebraic loop."""

        g = Graph(self.nodes_alg_loop, self.conns_alg_loop)
        alg_depth, loop_depth = g.depth()

        # Upstream path length is None for blocks in loop -> _blocks_dag empty -> alg_depth = 0
        # Loop BFS: add2 (entry depth 0) -> amp1 (depth 1). Max loop depth = 1. Graph._loop_depth = max + 1 = 2.
        self.assertEqual(alg_depth, 0) 
        self.assertEqual(loop_depth, 2) 


    def test_depth_dynamic_loop(self):
        """Test depth calculation with only a dynamic loop."""

        g = Graph(self.nodes_dyn_loop, self.conns_dyn_loop)

        alg_depth, loop_depth = g.depth()

        # Upstream path length for int1/int2 is 0 (stops at non-algebraic block). Max depth = 0.
        # Graph._alg_depth = max + 1 = 1.
        # No algebraic loop -> loop_depth = 0.
        self.assertEqual(alg_depth, 1) 
        self.assertEqual(loop_depth, 0) 


    def test_dag_traversal_mixed(self):
        """Test DAG traversal on a mixed acyclic graph."""

        g = Graph(self.nodes_acyclic, self.conns_acyclic)

        dag_result = list(g.dag())

        self.assertEqual(len(dag_result), 4) # Graph depth is 4, so levels 0, 1, 2, 3 exist

        # Check Depth 0
        self.assertIn(self.int1, dag_result[0][1])
        self.assertEqual(len(dag_result[0][1]), 2)
        self.assertEqual(len(dag_result[0][2]), 1) 

        # Check Depth 1
        self.assertIn(self.amp1, dag_result[1][1])
        self.assertEqual(len(dag_result[1][1]), 1)
        self.assertIn(self.c_int1_add1, dag_result[0][2]) # Connection from depth 0 node (int1)

        # Check Depth 2
        self.assertIn(self.add1, dag_result[2][1])
        self.assertEqual(len(dag_result[2][1]), 1)
        self.assertIn(self.c_amp1_add1, dag_result[1][2]) # Connection from depth 1 node (amp1)

        # Check Depth 3
        self.assertIn(self.amp2, dag_result[3][1])
        self.assertIn(self.int2, dag_result[0][1])
        self.assertEqual(len(dag_result[3][1]), 1)
        self.assertIn(self.c_add1_amp2, dag_result[2][2]) # Connection from depth 2 node (add1)
        self.assertIn(self.c_amp2_int2, dag_result[3][2]) # Connection from depth 3 node (amp2) - yields at source depth



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)