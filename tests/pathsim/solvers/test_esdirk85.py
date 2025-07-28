########################################################################################
##
##                                  TESTS FOR 
##                             'solvers/esdirk85.py'
##
##                               Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.esdirk85 import ESDIRK85

from tests.pathsim.solvers._referenceproblems import PROBLEMS

import matplotlib.pyplot as plt


# TESTS ================================================================================

class TestESDIRK85(unittest.TestCase):
    """
    Test the implementation of the 'ESDIRK85' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = ESDIRK85()

        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = ESDIRK85(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)



    def test_stages(self):

        solver = ESDIRK85()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_step(self):

        solver = ESDIRK85()
        
        solver.buffer(1)

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            _ = solver.solve(0.0, 0.0, 1) #needed for implicit solvers to get slope
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

        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(0.6, 1.2, 20)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = ESDIRK85(problem.x0)
                
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
                        adaptive=False,
                        tolerance_fpi=1e-12,
                        max_iterations=1000
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                #test if errors are monotonically decreasing (doesnt make sense because not smooth)
                # self.assertTrue(np.all(np.diff(errors)<0))

                #test convergence order, expected n-1 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, solver.n-2)

            #log stats
            stats[problem.name] = {"n":p, "err":errors, "dt":timesteps}

        # fig, ax = plt.subplots(dpi=120, tight_layout=True)
        # fig.suptitle(solver.__class__.__name__)
        # for name, stat in stats.items(): 
        #     ax.loglog(stat["dt"], stat["err"], label=name)
        # ax.loglog(timesteps, timesteps**solver.n, c="k", ls="--", label=f"n={solver.n}")
        # ax.legend()
        # plt.show()


    def test_integrate_adaptive(self):

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = ESDIRK85(problem.x0, tolerance_lte_rel=0, tolerance_lte_abs=1e-5)

                duration = problem.t_span[1] - problem.t_span[0]
                
                time, numerical_solution = solver.integrate(
                    problem.func, 
                    problem.jac,
                    time_start=problem.t_span[0], 
                    time_end=problem.t_span[1], 
                    dt=duration/100, 
                    dt_max=duration,
                    adaptive=True,
                    tolerance_fpi=1e-8
                    )

                analytical_solution = problem.solution(time)
                err = np.mean(abs(numerical_solution - analytical_solution))

                #test if error control was successful (same OOM for global error -> < 1e-5)
                self.assertLess(err, solver.tolerance_lte_abs*10)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':

    unittest.main(verbosity=2)