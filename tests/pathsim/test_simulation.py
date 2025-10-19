########################################################################################
##
##                                  TESTS FOR 
##                               'simulation.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.simulation import Simulation

#for testing
from pathsim.blocks._block import Block
from pathsim.connection import Connection

#modules from pathsim for test case
from pathsim.blocks import (
    Integrator,
    Amplifier,  
    Scope, 
    Adder
    )

from pathsim._constants import (
    SIM_TIMESTEP,
    SIM_TIMESTEP_MIN,
    SIM_TIMESTEP_MAX,
    SIM_TOLERANCE_FPI,
    SIM_ITERATIONS_MAX
    )


# TESTS ================================================================================

class TestSimulation(unittest.TestCase):
    """
    Test the implementation of the 'Simulation' class

    only very minimal functonality
    """

    def test_init_default(self):

        #test default initialization
        Sim = Simulation(log=False)
        self.assertEqual(Sim.blocks, set())
        self.assertEqual(Sim.connections, set())
        self.assertEqual(Sim.events, set())
        self.assertEqual(Sim.dt, SIM_TIMESTEP)
        self.assertEqual(Sim.dt_min, SIM_TIMESTEP_MIN)
        self.assertEqual(Sim.dt_max, SIM_TIMESTEP_MAX)
        self.assertEqual(str(Sim.Solver()), "SSPRK22")
        self.assertEqual(Sim.tolerance_fpi, SIM_TOLERANCE_FPI)
        self.assertEqual(Sim.iterations_max, SIM_ITERATIONS_MAX)
        self.assertFalse(Sim.log)


    def test_init_sepecific(self):

        #test specific initialization
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3], 
            dt=0.02, 
            dt_min=0.001, 
            dt_max=0.1, 
            tolerance_fpi=1e-9, 
            tolerance_lte_rel=1e-4, 
            tolerance_lte_abs=1e-6, 
            iterations_max=100, 
            log=False
            )
        self.assertEqual(len(Sim.blocks), 3)
        self.assertEqual(len(Sim.connections), 3)
        self.assertEqual(Sim.dt, 0.02)
        self.assertEqual(Sim.dt_min, 0.001)
        self.assertEqual(Sim.dt_max, 0.1)
        self.assertEqual(Sim.tolerance_fpi, 1e-9)
        self.assertEqual(Sim.solver_kwargs, {"tolerance_lte_rel":1e-4, "tolerance_lte_abs":1e-6})
        self.assertEqual(Sim.iterations_max, 100)

        #test specific initialization with connection override
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B2) # <-- overrides B2
        with self.assertRaises(ValueError):
            Sim = Simulation(
                blocks=[B1, B2, B3], 
                connections=[C1, C2, C3],
                log=False
                )


    def test_contains(self):

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C3],
            log=False
            )

        self.assertTrue(B1 in Sim)
        self.assertTrue(B2 in Sim)
        self.assertTrue(B3 in Sim)

        self.assertTrue(C1 in Sim)
        self.assertTrue(C2 not in Sim)
        self.assertTrue(C3 in Sim)


    def test_add_block(self):
        
        Sim = Simulation(log=False)

        self.assertEqual(Sim.blocks, set())

        #test adding a block
        B1 = Block()
        Sim.add_block(B1)
        self.assertEqual(Sim.blocks, {B1})

        #test adding the same block again
        with self.assertRaises(ValueError):
            Sim.add_block(B1)


    def test_add_connection(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)

        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1],
            log=False
            )

        self.assertEqual(Sim.connections, {C1})

        #test adding a connection
        C2 = Connection(B2, B3)
        Sim.add_connection(C2)
        self.assertEqual(Sim.connections, {C1, C2})

        #test adding the same connection again
        with self.assertRaises(ValueError):
            Sim.add_connection(C2)
        self.assertEqual(Sim.connections, {C1, C2})


    def test_set_solver(self): 

        from pathsim.solvers import SSPRK22, RKCK54
        

        B1, B2 = Block(), Block()
        I1, I2 = Integrator(), Integrator()
        C1 = Connection(B1, B2)

        #check no solvers yet
        self.assertEqual(I1.engine, None)
        self.assertEqual(I2.engine, None)
        self.assertEqual(B1.engine, None)
        self.assertEqual(B2.engine, None)

        Sim = Simulation(
            blocks=[B1, B2, I1, I2], 
            connections=[C1],
            log=False
            )

        #check solvers initialized correctly
        self.assertTrue(isinstance(I1.engine, SSPRK22))
        self.assertTrue(isinstance(I2.engine, SSPRK22))
        self.assertEqual(B1.engine, None)
        self.assertEqual(B2.engine, None)

        Sim._set_solver(RKCK54)

        #check solvers are correctly updated
        self.assertTrue(isinstance(I1.engine, RKCK54))
        self.assertTrue(isinstance(I2.engine, RKCK54))
        self.assertEqual(B1.engine, None)
        self.assertEqual(B2.engine, None)


    def test_size(self):    

        #test 3 alg. blocks
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3],
            log=False
            )  

        n, nx = Sim.size
        self.assertEqual(n, 3)
        self.assertEqual(nx, 0)

        #test 1 dyn, 1 alg block
        B1, B2 = Block(), Integrator()
        C1 = Connection(B1, B2)
        Sim = Simulation(
            blocks=[B1, B2], 
            connections=[C1],
            log=False
            )  

        n, nx = Sim.size
        self.assertEqual(n, 2)
        self.assertEqual(nx, 1)


    def test_update(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3],
            log=False
            )   

        Sim._update(1)


    def test_step(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3],
            log=False
            )   

        self.assertEqual(Sim.time, 0.0)

        #check stepping stats
        suc, err, scl, evl, its = Sim.step(0.1)

        self.assertTrue(suc)
        self.assertEqual(err, 0.0)
        self.assertEqual(scl, 1.0)
        self.assertEqual(evl, 1)
        self.assertEqual(its, 0)
        self.assertEqual(Sim.time, 0.1)

        Sim.reset()
        self.assertEqual(Sim.time, 0.0)
            
        #test time progression
        for i in range(1, 10):

            stats = Sim.step(0.1)
            self.assertAlmostEqual(Sim.time, 0.1*i, 10)


    def test_stop(self):
        """Test that stop() method sets _active flag to False"""
        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1],
            log=False
        )
        
        # Initially active
        self.assertTrue(Sim._active)
        self.assertTrue(bool(Sim))  # __bool__ should return _active
        
        # Stop the simulation
        Sim.stop()
        
        # Should be inactive
        self.assertFalse(Sim._active)
        self.assertFalse(bool(Sim))


    def test_run(self): 

        B1, B2, B3 = Block(), Block(), Block()
        C1 = Connection(B1, B2)
        C2 = Connection(B2, B3)
        C3 = Connection(B3, B1)
        Sim = Simulation(
            blocks=[B1, B2, B3], 
            connections=[C1, C2, C3],
            log=False
            )   

        self.assertEqual(Sim.time, 0.0)

        stats = Sim.run(10)

        #check stats
        self.assertEqual(stats["total_steps"], 1001)
            
        #check internal time progression (fixed step solvers naturally overshoot)
        self.assertAlmostEqual(Sim.time, 10, 1)





class TestSimulationIVP(unittest.TestCase):
    """
    special test case:
    linear feedback initial value problem with default solver (SSPRK22)
    """

    def setUp(self):

        #blocks that define the system
        self.Int = Integrator(1.0)
        self.Amp = Amplifier(-1)
        self.Add = Adder()
        self.Sco = Scope(labels=["response"])

        blocks = [self.Int, self.Amp, self.Add, self.Sco]

        #the connections between the blocks
        connections = [
            Connection(self.Amp, self.Add[1]),
            Connection(self.Add, self.Int),
            Connection(self.Int, self.Amp, self.Sco)
            ]

        #initialize simulation with the blocks, connections, timestep and logging enabled
        self.Sim = Simulation(blocks, connections, dt=0.02, log=False)


    def test_init(self):

        from pathsim.solvers import SSPRK22

        #test initialization of simulation
        self.assertEqual(len(self.Sim.blocks), 4)
        self.assertEqual(len(self.Sim.connections), 3)
        self.assertEqual(self.Sim.dt, 0.02)
        self.assertTrue(isinstance(self.Sim.engine, SSPRK22))
        self.assertTrue(self.Sim.engine.is_explicit)
        self.assertFalse(self.Sim.engine.is_adaptive)

        #test if engine setup was correct
        self.assertTrue(isinstance(self.Int.engine, SSPRK22)) # <-- only the Integrator needs an engine
        self.assertTrue(self.Amp.engine is None)
        self.assertTrue(self.Add.engine is None)
        self.assertTrue(self.Sco.engine is None)


    def test_set_solver(self):

        from pathsim.solvers import BDF2, SSPRK22

        #reset first
        self.Sim.reset()
    
        #set to implicit solver
        self.Sim._set_solver(BDF2)
        self.assertTrue(isinstance(self.Sim.engine, BDF2))

        #run with implicit solver
        self.test_run()

        #set to explicit solver
        self.Sim._set_solver(SSPRK22)
        self.assertTrue(isinstance(self.Sim.engine, SSPRK22))

        #run with explicit solver
        self.test_run()


    def test_step(self):

        #reset first
        self.Sim.reset()

        #check if reset was sucecssful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.outputs[0], self.Int.initial_value)

        #step using global timestep
        success, err, scl, te, ts = self.Sim.timestep()
        self.assertEqual(self.Sim.time, self.Sim.dt)
        self.assertEqual(err, 0.0) #fixed solver
        self.assertEqual(scl, 1.0) #fixed solver
        self.assertEqual(ts, 0) #no implicit solver
        
        #step again using custom timestep
        self.Sim.step(dt=2.2*self.Sim.dt)
        self.assertEqual(self.Sim.time, 3.2*self.Sim.dt)
        self.assertLess(self.Int.outputs[0], self.Int.initial_value)

        #test if scope recorded correctly
        time, data = self.Sco.read()
        for a, b in zip(time, [self.Sim.dt, 3.2*self.Sim.dt]):
            self.assertEqual(a, b)

        #reset again
        self.Sim.reset()

        #check if reset was successful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.outputs[0], self.Int.initial_value)


    def test_run(self):

        #reset first
        self.Sim.reset()

        #check if reset was successful
        self.assertEqual(self.Sim.time, 0.0)
        self.assertEqual(self.Int.outputs[0], self.Int.initial_value)

        #test running for some time
        self.Sim.run(duration=2, reset=True)
        self.assertAlmostEqual(self.Sim.time, 2, 5)
        
        time, data = self.Sco.read()
        _time = np.arange(0, 2.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 
        self.assertTrue(np.all(np.diff(data) < 0.0))

        #test running for some time with reset
        self.Sim.run(duration=1, reset=True)
        self.assertAlmostEqual(self.Sim.time, 1, 5)

        time, data = self.Sco.read()
        _time = np.arange(0, 1.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 

        #test running for some time without reset
        self.Sim.run(duration=2, reset=False)
        self.assertAlmostEqual(self.Sim.time, 3, 5)

        time, data = self.Sco.read()
        _time = np.arange(0, 3.02, 0.02)

        #time recording matches and solution decays
        self.assertLess(np.linalg.norm(time - _time), 1e-13) 







# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)