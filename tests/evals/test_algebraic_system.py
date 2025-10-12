########################################################################################
##
##                          Testing algebraic paths system
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim import Simulation, Connection
from pathsim.blocks import Constant, Source, Function, Amplifier, Adder, Scope


# TESTCASE =============================================================================

class TestAlgebraicSystem(unittest.TestCase):

    def setUp(self):

        """Set up the system with everything thats needed 
        for the evaluation exposed"""

                
        #blocks that define the system
        Src = Source(np.sin)
        Cns = Constant(-1/2)
        Amp = Amplifier(2)
        Fnc = Function(lambda x: x**2)
        Add = Adder()
        self.Sco = Scope()

        blocks = [Src, Cns, Amp, Fnc, Add, self.Sco]

        #the connections between the blocks
        connections = [
            Connection(Src, Fnc, self.Sco),
            Connection(Fnc, Add[0], self.Sco[1]),
            Connection(Cns, Add[1]),
            Connection(Add, Amp),
            Connection(Amp, self.Sco[2])
            ]

        #initialize simulation with the blocks, connections
        self.Sim = Simulation(
            blocks, 
            connections, 
            dt=0.01,
            log=False
            )


    def test_graph(self):

        na, nd = self.Sim.size
        self.assertEqual(na, 6) # 6 alg. blocks
        self.assertEqual(nd, 0) # 0 dyn. blocks

        d_dag, d_loop = self.Sim.graph.depth
        self.assertEqual(d_dag, 4) # dag depth is 4
        self.assertEqual(d_loop, 0) # no alg. loops


    def test_eval(self):

        #reference solution of alg. system
        def ref(t):
            return np.sin(t), np.sin(t)**2, -np.cos(2*t) 

        self.Sim.run(20)

        time, [a, b, c] = self.Sco.read()
        ar, br, cr = ref(time)

        self.assertTrue(np.allclose(a, ar))
        self.assertTrue(np.allclose(b, br))
        self.assertTrue(np.allclose(c, cr))


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
