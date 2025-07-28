########################################################################################
##
##                                  TESTS FOR 
##                              'solvers/gear.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.gear import *

from tests.pathsim.solvers._referenceproblems import PROBLEMS

import matplotlib.pyplot as plt


# TESTS ================================================================================

class TestComputeBDFCoefficients(unittest.TestCase):
    """
    Test the implementation of 'compute_bdf_coefficients'
    """

    def test_order_1(self):

        n = 1

        F, K = 1.0, [1.0]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_2(self):

        n = 2

        F, K = 2/3, [-1/3, 4/3]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_3(self):

        n = 3

        F, K = 6/11, [2/11, -9/11, 18/11]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_4(self):

        n = 4

        F, K = 12/25, [-3/25, 16/25, -36/25, 48/25]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_5(self):

        n = 5

        F, K = 60/137, [12/137, -75/137, 200/137, -300/137, 300/137]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


    def test_order_6(self):

        n = 6

        F, K = 60/147, [-10/147, 72/147, -225/147, 400/147, -450/147, 360/147]

        _F, _K = compute_bdf_coefficients(n, np.ones(n))

        #test if bdf coefficients for fixed timestep are computed correctly
        self.assertAlmostEqual(_F, F, 7)
        for _k, k in zip(_K, K[::-1]):
            self.assertAlmostEqual(_k, k, 7)


# TESTS FOR SPECIFIC GEAR TYPE SOLVERS =================================================


class TestGEAR21(unittest.TestCase):
    """
    Test the implementation of the 'GEAR21' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = GEAR21()

        self.assertEqual(solver.initial_value, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = GEAR21(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = GEAR21()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(1)

            #test bdf buffer length
            self.assertEqual(len(solver.history), k+1 if k < solver.n else solver.n)
            self.assertEqual(len(solver.history_dt), k+1 if k < solver.n else solver.n)
            

    def test_integrate_fixed(self):

        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1.2, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = GEAR21(problem.x0)
                
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

                #test if errors are monotonically decreasing
                self.assertTrue(np.all(np.diff(errors)<0))

                #test convergence order, expected 2 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 1.5) # <- due to startup

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

                solver = GEAR21(problem.x0, tolerance_lte_rel=0, tolerance_lte_abs=1e-5)

                duration = problem.t_span[1] - problem.t_span[0]
                
                time, numerical_solution = solver.integrate(
                    problem.func, 
                    problem.jac,
                    time_start=problem.t_span[0], 
                    time_end=problem.t_span[1], 
                    dt_max=duration,
                    adaptive=True,
                    tolerance_fpi=1e-8
                    )

                analytical_solution = problem.solution(time)
                err = np.mean(numerical_solution - analytical_solution)

                #test if error control was successful (same OOM for global error -> < 1e-5)
                self.assertLess(err, solver.tolerance_lte_abs*10)



class TestGEAR32(unittest.TestCase):
    """
    Test the implementation of the 'GEAR32' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = GEAR32()

        self.assertEqual(solver.initial_value, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = GEAR32(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = GEAR32()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(1)

            #test bdf buffer length
            self.assertEqual(len(solver.history), k+1 if k < solver.n else solver.n)
            self.assertEqual(len(solver.history_dt), k+1 if k < solver.n else solver.n)


    def test_integrate_fixed(self):

        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1.2, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = GEAR32(problem.x0)
                
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

                #test if errors are monotonically decreasing
                self.assertTrue(np.all(np.diff(errors)<0))

                #test convergence order, expected 2 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 2) 


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

                solver = GEAR32(problem.x0, tolerance_lte_rel=0, tolerance_lte_abs=1e-5)

                duration = problem.t_span[1] - problem.t_span[0]
                
                time, numerical_solution = solver.integrate(
                    problem.func, 
                    problem.jac,
                    time_start=problem.t_span[0], 
                    time_end=problem.t_span[1], 
                    dt_max=duration,
                    adaptive=True,
                    tolerance_fpi=1e-8
                    )

                analytical_solution = problem.solution(time)
                err = np.mean(numerical_solution - analytical_solution)

                #test if error control was successful (same OOM for global error -> < 1e-5)
                self.assertLess(err, solver.tolerance_lte_abs*10)




class TestGEAR43(unittest.TestCase):
    """
    Test the implementation of the 'GEAR43' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = GEAR43()

        self.assertEqual(solver.initial_value, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = GEAR43(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = GEAR43()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(1)

            #test bdf buffer length
            self.assertEqual(len(solver.history), k+1 if k < solver.n else solver.n)
            self.assertEqual(len(solver.history_dt), k+1 if k < solver.n else solver.n)
            

    def test_integrate_fixed(self):

        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = GEAR43(problem.x0)
                
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

                #test if errors are monotonically decreasing
                self.assertTrue(np.all(np.diff(errors)<0))

                #test convergence order, expected 3 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 3) 

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

                solver = GEAR43(problem.x0, tolerance_lte_rel=0, tolerance_lte_abs=1e-5)

                duration = problem.t_span[1] - problem.t_span[0]
                
                time, numerical_solution = solver.integrate(
                    problem.func, 
                    problem.jac,
                    time_start=problem.t_span[0], 
                    time_end=problem.t_span[1], 
                    dt_max=duration,
                    adaptive=True,
                    tolerance_fpi=1e-8
                    )

                analytical_solution = problem.solution(time)
                err = np.mean(numerical_solution - analytical_solution)

                #test if error control was successful (same OOM for global error -> < 1e-5)
                self.assertLess(err, solver.tolerance_lte_abs*10)



class TestGEAR54(unittest.TestCase):
    """
    Test the implementation of the 'GEAR54' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = GEAR54()

        self.assertEqual(solver.initial_value, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = GEAR54(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_buffer(self):

        solver = GEAR54()

        #perform some steps
        for k in range(10):

            #buffer state
            solver.buffer(1)

            #test bdf buffer length
            self.assertEqual(len(solver.history), k+1 if k < solver.n else solver.n)
            self.assertEqual(len(solver.history_dt), k+1 if k < solver.n else solver.n)
            

    def test_integrate_fixed(self):

        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = GEAR54(problem.x0)
                
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

                #test if errors are monotonically decreasing
                self.assertTrue(np.all(np.diff(errors)<0))

                #test convergence order, expected 3 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 3) # <- due to startup ESDIRK32

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

                solver = GEAR54(problem.x0, tolerance_lte_rel=0, tolerance_lte_abs=1e-5)

                duration = problem.t_span[1] - problem.t_span[0]
                
                time, numerical_solution = solver.integrate(
                    problem.func, 
                    problem.jac,
                    time_start=problem.t_span[0], 
                    time_end=problem.t_span[1], 
                    dt_max=duration,
                    adaptive=True,
                    tolerance_fpi=1e-8
                    )

                analytical_solution = problem.solution(time)
                err = np.mean(numerical_solution - analytical_solution)

                #test if error control was successful (same OOM for global error -> < 1e-5)
                self.assertLess(err, solver.tolerance_lte_abs*10)




class TestGEAR52A(unittest.TestCase):
    """
    Test the implementation of the 'GEAR52A' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = GEAR52A()

        self.assertEqual(solver.initial_value, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_implicit)
        self.assertFalse(solver.is_explicit)
        
        #test specific initialization
        solver = GEAR52A(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_integrate_fixed(self):

        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 30)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = GEAR52A(problem.x0)
                
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

                #test convergence order, expected >2 (global)
                p, _ = np.polyfit(np.log10(timesteps), np.log10(errors), deg=1)
                self.assertGreater(p, 2) 

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

                solver = GEAR52A(problem.x0, tolerance_lte_rel=0, tolerance_lte_abs=1e-5)

                duration = problem.t_span[1] - problem.t_span[0]
                
                time, numerical_solution = solver.integrate(
                    problem.func, 
                    problem.jac,
                    time_start=problem.t_span[0], 
                    time_end=problem.t_span[1], 
                    dt_max=duration,
                    adaptive=True,
                    tolerance_fpi=1e-8
                    )

                analytical_solution = problem.solution(time)
                err = np.mean(numerical_solution - analytical_solution)

                #test if error control was successful (same OOM for global error -> < 1e-5)
                self.assertLess(err, solver.tolerance_lte_abs*10)




# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':

    unittest.main(verbosity=2)
