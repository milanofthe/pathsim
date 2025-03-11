########################################################################################
##
##                                 UTILITY FUNCTIONS  
##                                  (utils/utils.py)
##
##                                Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

import numpy as np


# HELPERS FOR SIMULATION ===============================================================

def dict_to_array(a):
    """convert a dict with integer keys to a numpy array

    Parameters
    ----------
    a : dict[int: int, float, complex]
        dict to convert to numpy array
        
    Returns
    -------
    out : array[int, float, complex]
        converted array
    """
    return np.array([a[k] for k in sorted(a.keys())])


def array_to_dict(a):
    """convert a numpy array to a dict with integer keys

    Parameters
    ----------
    a : array[int, float, complex]
        numpy array to convert
    
    Returns
    -------
    out: dict[int: int, float, complex]
        converted dict
    """
    if np.isscalar(a): return {0:a}
    else: return dict(enumerate(a))


def rel_error(a, b):
    """Computes the relative error between two scalars.
    It is robust to one of them being zero and falls 
    back to the absolute error in this case.

    Notes
    ----- 
    this is actually faster then inlining the 
    branching into the return statement
    
    Parameters
    ----------
    a : float, int, complex
        first number
    b : float, int, complex
        second number
    
    Returns
    -------
    err : float
        retative error
    """
    if a == 0.0: return abs(b)
    else: return abs((a - b)/a)


def abs_error(a, b):
    """Computes the absolute error between two scalars.

    Parameters
    ----------
    a : float, int, complex
        first number
    b : float, int, complex
        second number
    
    Returns
    -------
    err : float
        absolute error
    """
    return abs(a - b)


def max_error(a, b):
    """Computes the maximum absolute error / deviation between two
    iterables such as lists with numerical values. Returns a scalar 
    value representing the maximum deviation.

    Notes
    -----
    this is actually faster then 'max' over a list comprehension
    
    Parameters
    ----------
    a : iterable[float, int, complex]
        first iterable with numerical values
    b : iterable[float, int, complex]
        second iterable with numerical values
    
    Returns
    -------
    err : float
        maximum absolute error
    """
    max_err = 0.0
    for err in map(abs_error, a, b):
        if err > max_err: 
            max_err = err
    return max_err


def max_rel_error(a, b):
    """Computes the maximum relative error between two iterables 
    such as lists with numerical values. 

    It is robust to one of them being zero and falls back to the 
    absolute error in this case. 

    It returns a scalar value representing the maximum relative error. 

    Notes
    -----
    this is actually faster then 'max' over a list comprehension

    Parameters
    ----------
    a : iterable[float, int, complex]
        first iterable with numerical values
    b : iterable[float, int, complex]
        second iterable with numerical values
    
    Returns
    -------
    err : float
        maximum retative error
    """
    max_err = 0.0
    for err in map(rel_error, a, b):
        if err > max_err: 
            max_err = err
    return max_err


def max_error_dicts(a, b):
    """Computes the maximum absolute error between two dictionaries 
    with numerical values. 

    It returns a scalar value representing the maximum absolute error.

    Parameters
    ----------
    a : dict[int: float, int, complex]
        first dict with numerical values
    b : dict[int: float, int, complex]
        second iterable with numerical values

    Returns
    -------
    err : float
        maximum absolute error
    """
    return max_error(a.values(), b.values())


def max_rel_error_dicts(a, b):
    """Computes the maximum relative error between two dictionaries 
    with numerical values. 

    It is robust to one of them being zero and falls back to the 
    absolute error in this case. 

    It returns a scalar value representing the maximum relative error. 

    Parameters
    ----------
    a : dict[int: float, int, complex]
        first dict with numerical values
    b : dict[int: float, int, complex]
        second iterable with numerical values

    Returns
    -------
    err : float
        maximum relative error
    """
    return max_rel_error(a.values(), b.values())


# PATH ESTIMATION ======================================================================

def path_length_dfs(connections, starting_block, visited=None):
    """Recursively compute the longest path (depth first search) 
    in a directed graph from a starting node / block.
    
    Parameters
    ----------
    connections : list[Connection]
        connections of the graph
    starting_block : Block
        block to start dfs
    visited : None, set
        set of already visited graph nodes (blocks)
    
    Returns
    -------
    length : int
        length of path starting from ´starting_block´
    """

    if visited is None:
        visited = set()

    #node already visited -> break cycles
    if starting_block in visited:
        return 0

    #block without instant time component -> break cycles
    if not len(starting_block):   
        return 0

    #add starting node to set of visited nodes
    visited.add(starting_block)

    #length of paths from the starting nodes
    max_length = 0

    #iterate connections and explore the path from the target node
    for conn in connections:
        
        #find connections from starting block
        src, _ = conn.source 
        if src == starting_block:

            #iterate connection target blocks
            for trg, _ in conn.targets:

                #recursively compute the new longest path
                length = path_length_dfs(connections, trg, visited.copy())
                if length > max_length: max_length = length

    #add the contribution of the starting node to longest path
    return max_length + len(starting_block)

