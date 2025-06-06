########################################################################################
##
##                                  TESTS FOR 
##                              'solvers/dirk2.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.dirk2 import DIRK2

from tests.pathsim.solvers._referenceproblems import PROBLEMS


# TESTS ================================================================================

class TestDIRK2(unittest.TestCase):
    """
    Test the implementation of the 'DIRK2' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = DIRK2()

        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertFalse(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = DIRK2(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_stages(self):

        solver = DIRK2()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_step(self):

        solver = DIRK2()

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            _ = solver.solve(0.0, 0.0, 1) #needed for implicit solvers to get slope
            success, err, scale = solver.step(0.0, 1)

            #test if expected return at intermediate stages
            self.assertTrue(success)
            self.assertEqual(err, 0.0)
            self.assertEqual(scale, 1.0)

    
    def test_integrate_fixed(self):
        
        #divisons of integration duration
        divisions = np.logspace(1.5, 2.5, 20)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = DIRK2(problem.x0)
                
                errors = []

                timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

                for dt in timesteps:

                    solver.reset()
                    time, numerical_solution = solver.integrate(
                        problem.func, 
                        problem.jac,
                        time_start=problem.t_span[0], 
                        time_end=problem.t_span[1], 
                        dt=dt, 
                        adaptive=False
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                #test convergence order, expected n-1 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, solver.n-1)





# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':

    unittest.main(verbosity=2)