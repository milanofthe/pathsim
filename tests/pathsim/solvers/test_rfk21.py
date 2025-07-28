########################################################################################
##
##                                  TESTS FOR 
##                             'solvers/rkf21.py'
##
##                              Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.rkf21 import RKF21

from tests.pathsim.solvers._referenceproblems import PROBLEMS


# TESTS ================================================================================

class TestRKF21(unittest.TestCase):
    """
    Test the implementation of the 'RKF21' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = RKF21()

        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_explicit)
        self.assertFalse(solver.is_implicit)
        
        #test specific initialization
        solver = RKF21(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_stages(self):

        solver = RKF21()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_step(self):

        solver = RKF21()
        
        solver.buffer(1)

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            success, err, scale = solver.step(0.0, 1)

            #test if expected return at intermediate stages
            if i < len(solver.eval_stages)-1:
                self.assertTrue(success)
                self.assertEqual(err, 0.0)
                self.assertEqual(scale, 1.0)

        #test if expected return at final stage
        self.assertNotEqual(err, 0.0)
        self.assertNotEqual(scale, 1.0)

    
    def test_integrate_fixed(self):
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = RKF21(problem.x0)
                
                errors = []

                timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

                for dt in timesteps:

                    solver.reset()
                    time, numerical_solution = solver.integrate(
                        problem.func, 
                        time_start=problem.t_span[0], 
                        time_end=problem.t_span[1], 
                        dt=dt, 
                        adaptive=False
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                #test if errors are monotonically decreasing
                self.assertTrue(np.all(np.diff(errors)<0))

                #test convergence order, expected n-1 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, solver.n-1)


    def test_integrate_adaptive(self):

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = RKF21(problem.x0, tolerance_lte_rel=0, tolerance_lte_abs=1e-4)

                duration = problem.t_span[1] - problem.t_span[0]
                
                time, numerical_solution = solver.integrate(
                    problem.func, 
                    time_start=problem.t_span[0], 
                    time_end=problem.t_span[1], 
                    dt=duration/100, 
                    adaptive=True
                    )

                analytical_solution = problem.solution(time)
                err = np.mean(numerical_solution - analytical_solution)

                #test if error control was successful (same OOM for global error -> lte < 1e-5)
                self.assertLess(err, solver.tolerance_lte_abs*50)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':

    unittest.main(verbosity=2)