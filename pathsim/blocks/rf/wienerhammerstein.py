#########################################################################################
##
##                    WIENER AND HAMMERSTEIN NONLINEAR DYNAMICAL MODELS 
##                             (blocks/wienerhammerstein.py)
##
##                                  Milan Rother 2024
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np

from .._block import Block

from ...utils.funcs import (
    max_error_dicts,
    dict_to_array, 
    array_to_dict,
    auto_jacobian
    )

from ...utils.gilbert import (
    gilbert_realization
    )


# BLOCKS ================================================================================

class WienerHammersteinModel(Block):
    """
    Wiener-Hammerstein nonlinear dynamical model. 
    It consists of a dynamical LTI system, followed 
    by a static nonlinearity, followed by another 
    dynamical LTI system.
    
        u -> LTI_1 -> f(.) -> LTI_2 -> y 

    The LTI systems are implemented as linear statespace 
    models with ABCD system matrices. 
    The system matrices are realized from the poles, residues 
    and constant of the corresponding transfer functions. 
    
    Can serve as a behavioral model for nonlinear 
    dynamical systems such as RF components. 

    This implementation is inherently MIMO-capable 
    if the IO dimensions of the two LTI systems and 
    the nonlinearity match.

    INPUTS : 
        Poles_1    : (array of complex) poles of 1st LTI system transfer function
        Residues_1 : (array of arrays) residues of 1st LTI system transfer function
        Const_1    : (array of arrays) constant term of 1st LTI system transfer function
        Poles_2    : (array of complex) poles of 2nd LTI system transfer function
        Residues_2 : (array of arrays) residues of 2nd LTI system transfer function
        Const_2    : (array of arrays) constant term of 2nd LTI system transfer function
        func       : (callable) function that defines the system nonlienarity, can be MIMO 

    """

    def __init__(self, 
                 Poles_1=[], 
                 Residues_1=[], 
                 Const_1=1.0,
                 Poles_2=[], 
                 Residues_2=[], 
                 Const_2=1.0,
                 func=lambda x: x,
                 jac=None):
        
        super().__init__()

        #nonlinearity
        self.func = func

        #jacobian of nonlinearity
        self.jac = auto_jacobian(self.func) if jac is None else jac

        #Statespace realization of 1st LTI transfer function
        self.A_1, self.B_1, self.C_1, self.D_1 = gilbert_realization(Poles_1, Residues_1, Const_1)

        #Statespace realization of 2nd LTI transfer function
        self.A_2, self.B_2, self.C_2, self.D_2 = gilbert_realization(Poles_2, Residues_2, Const_2)

        #get statespace dimensions after realization 
        _, n_in  = self.B_1.shape 
        n_out, _ = self.C_2.shape 

        self.n_1, _ = self.A_1.shape
        self.n_2, _ = self.A_2.shape

        #set io channels
        self.inputs  = {i:0.0 for i in range(n_in)}
        self.outputs = {i:0.0 for i in range(n_out)}


    def __len__(self):
        #check if direct passthrough exists
        return int((np.any(self.D_1) and np.any(self.D_2)))


    def set_solver(self, Solver, **solver_args):

        #change solver if already initialized
        if self.engine is not None:
            self.engine = self.engine.change(Solver, **solver_args)
            return #quit early

        #right hand side function for ODE
        def _f(x, u, t):
            x_1, x_2 = x[:self.n_1], x[self.n_1:]
            dx_1 = np.dot(self.A_1, x_1) + np.dot(self.B_1, u) 
            dx_2 = np.dot(self.A_2, x_2) + np.dot(self.B_2, self.func(np.dot(self.C_1, x_1) + np.dot(self.D_1, u)))
            return np.hstack([dx_1, dx_2])

        def _jac(x, u, t):
            x_1, x_2 = x[:self.n_1], x[self.n_1:]
            J_12 = np.zeros_like(self.A_1)
            J_21 = np.dot(self.B_2, np.dot(self.C_1, self.jac(np.dot(self.C_1, x_1) + np.dot(self.D_1, u))))
            return np.block([[self.A_1, J_12], [J_21, self.A_2]])
        
        #combined solver
        self.engine = Solver(np.zeros(self.n_1+self.n_2), _f, _jac, **solver_args)


    def update(self, t):
        #compute implicit balancing update
        prev_outputs = self.outputs.copy()

        x = self.engine.get()
        x_1, x_2 = x[:self.n_1], x[self.n_1:]
        
        u = dict_to_array(self.inputs)

        y_1 = np.dot(self.C_1, x_1) + np.dot(self.D_1, u)
        y_2 = np.dot(self.C_2, x_2) + np.dot(self.D_2, self.func(y_1))
        
        self.outputs = array_to_dict(y_2)
        return max_error_dicts(prev_outputs, self.outputs)


    def solve(self, t, dt):
        #advance solution of implicit update equation
        return self.engine.solve(dict_to_array(self.inputs), t, dt)


    def step(self, t, dt):
        #compute update step with integration engine
        return self.engine.step(dict_to_array(self.inputs), t, dt)






class HammersteinModel(Block):
    """
    Hammerstein nonlinear dynamical model. It consists of a 
    dynamical LTI system, followed by a static nonlinearity.
    
        u -> f(.) -> LTI -> y 

    The LTI system is implemented as a linear statespace 
    model with ABCD system matrices. 
    The system matrices are realized from the poles, residues 
    and constant of the corresponding transfer function. 
    
    Can serve as a behavioral model for nonlinear 
    dynamical systems such as RF components. 

    This implementation is inherently MIMO-capable 
    if the IO dimensions of the LTI system and 
    the nonlinearity match.

    INPUTS : 
        Poles    : (array of complex) poles of LTI system transfer function
        Residues : (array of arrays) residues of LTI system transfer function
        Const    : (array of arrays) constant term of LTI system transfer function
        func     : (callable) function that defines the system nonlienarity, can be MIMO 
    """
    
    def __init__(self, 
                 Poles=[], 
                 Residues=[], 
                 Const=1.0,
                 func=lambda x: x):
        
        super().__init__()

        #nonlinearity
        self.func = func

        #Statespace realization of transfer function
        self.A, self.B, self.C, self.D = gilbert_realization(Poles, Residues, Const)

        #get statespace dimensions after realization 
        _, n_in  = self.B.shape 
        n_out, _ = self.C.shape 

        #set io channels
        self.inputs  = {i:0.0 for i in range(n_in)}
        self.outputs = {i:0.0 for i in range(n_out)}


    def __len__(self):
        #check if direct passthrough exists
        return int(np.any(self.D))


    def set_solver(self, Solver, **solver_args):
        
        #change solver if already initialized
        if self.engine is not None:
            self.engine = self.engine.change(Solver, **solver_args)
            return #quit early

        #right hand side function for ODE
        def _f(x, u, t): return np.dot(self.A, x) + np.dot(self.B, u) 
        def _jac(x, u, t): return self.A
        
        #solver
        n, _ = self.A.shape
        self.engine = Solver(np.zeros(n), _f, _jac, **solver_args)


    def update(self, t):
        #compute implicit balancing update
        prev_outputs = self.outputs.copy()
        u = dict_to_array(self.inputs)
        y = np.dot(self.C, self.engine.get()) + np.dot(self.D, self.func(u))
        self.outputs = array_to_dict(y)
        return max_error_dicts(prev_outputs, self.outputs)

    def solve(self, t, dt):
        #advance solution of implicit update equation
        u = dict_to_array(self.inputs)
        return self.engine.solve(self.func(u), t, dt)

    def step(self, t, dt):
        #compute update step with integration engine
        u = dict_to_array(self.inputs)
        return self.engine.step(self.func(u), t, dt)







class WienerModel(Block):

    """
    Wiener nonlinear dynamical model. It consists of 
    a static nonlinearity, followed by a dynamical LTI system.
    
        u -> LTI -> f(.) -> y 

    The LTI system is implemented as a linear statespace 
    model with ABCD system matrices. 
    The system matrices are realized from the poles, residues 
    and constant of the corresponding transfer function. 
    
    Can serve as a behavioral model for nonlinear 
    dynamical systems such as RF components. 

    This implementation is inherently MIMO-capable 
    if the IO dimensions of the LTI system and 
    the nonlinearity match.

    INPUTS : 
        Poles    : (array of complex) poles of LTI system transfer function
        Residues : (array of arrays) residues of LTI system transfer function
        Const    : (array of arrays) constant term of LTI system transfer function
        func     : (callable) function that defines the system nonlienarity, can be MIMO 
    """
    
    def __init__(self, 
                 b=None, 
                 a=None, 
                 Poles=[], 
                 Residues=[], 
                 Const=1.0,
                 func=lambda x: x):
        
        super().__init__()

        #nonlinearity
        self.func = func

        #Statespace realization of transfer function
        self.A, self.B, self.C, self.D = gilbert_realization(Poles, Residues, Const)

        #get statespace dimensions after realization 
        _, n_in  = self.B.shape 
        n_out, _ = self.C.shape 

        #set io channels
        self.inputs  = {i:0.0 for i in range(n_in)}
        self.outputs = {i:0.0 for i in range(n_out)}

    def __len__(self):
        #check if direct passthrough exists
        return int(np.any(self.D))


    def set_solver(self, Solver, **solver_args):
        
        #change solver if already initialized
        if self.engine is not None:
            self.engine = self.engine.change(Solver, **solver_args)
            return #quit early

        #right hand side function for ODE
        def _f(x, u, t): return np.dot(self.A, x) + np.dot(self.B, u) 
        def _jac(x, u, t): return self.A 

        #solver
        n, _ = self.A.shape
        self.engine = Solver(np.zeros(n), _f, _jac, **solver_args)


    def update(self, t):
        #compute implicit balancing update
        prev_outputs = self.outputs.copy()
        y = np.dot(self.C, self.engine.get()) + np.dot(self.D, dict_to_array(self.inputs))
        self.outputs = array_to_dict(self.func(y))
        return max_error_dicts(prev_outputs, self.outputs)


    def solve(self, t, dt):
        #advance solution of implicit update equation
        return self.engine.solve(dict_to_array(self.inputs), t, dt)


    def step(self, t, dt):
        #compute update step with integration engine
        return self.engine.step(dict_to_array(self.inputs), t, dt)