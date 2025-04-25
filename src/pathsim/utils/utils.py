########################################################################################
##
##                                 UTILITY FUNCTIONS  
##                                  (utils/utils.py)
##
##                                Milan Rother 2023/24
##
########################################################################################

# IMPORTS ==============================================================================

# no dependencies


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
        if conn.source.block == starting_block:

            #iterate connection target blocks
            for trg in conn.targets:

                #recursively compute the new longest path
                length = path_length_dfs(connections, trg.block, visited.copy())
                if length > max_length: max_length = length

    #add the contribution of the starting node to longest path
    return max_length + len(starting_block)

