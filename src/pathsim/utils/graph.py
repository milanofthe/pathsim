########################################################################################
##
##                            FUNCTIONS FOR GRAPH ANALYSIS
##                                  (utils/graph.py)
##
##                                 Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

from collections import defaultdict, deque, namedtuple
from functools import cache


# HELPER METHODS =======================================================================

def outgoing_block_connection_map(connections):
    """Construct a mapping from blocks to their outgoing connections.

    Parameters
    ----------
    connections : list[Connection]
        connections of the graph

    Returns
    -------
    block_connection_map : defaultdict[Block, list[Connection]]
        outgoing connections of block
    """
    block_connection_map = defaultdict(list)
    for con in connections:
        src_blk = con.source.block
        block_connection_map[src_blk].append(con)
    return block_connection_map


def downstream_block_block_map(connections):
    """Construct a connection map (directed graph) from the list of
    connections in downstream (source -> targets) orientation.

    Parameters
    ----------
    connections : list[Connection]
        connections of the graph

    Returns
    -------
    connection_map : defaultdict[Block, set[Block]]
        directed downstream graph of connections (source -> targets)
    """
    connection_map = defaultdict(set)
    for con in connections:
        src_blk = con.source.block
        for trg in con.targets:
            connection_map[src_blk].add(trg.block)
    return connection_map


def upstream_block_block_map(connections):
    """Construct a connection map (directed graph) from the list of
    connections in upstream (targets -> source) orientation.

    Parameters
    ----------
    connections : list[Connection]
        connections of the graph

    Returns
    -------
    connection_map : defaultdict[Block, set[Block]]
        directed upstream graph of connections (target -> source)
    """
    connection_map = defaultdict(set)
    for con in connections:
        src_blk = con.source.block
        for trg in con.targets:
            connection_map[trg.block].add(src_blk)
    return connection_map


def algebraic_depth_dfs(graph_map, node, node_set=None, propagate_inf=True, stack=frozenset()):
    """Computes the longest algebraic path length using depth-first search.
    
    This function determines the longest algebraic path length from a given node,
    traversing the graph in the direction specified by the graph_map. It can
    operate in two modes controlled by the propagate_inf parameter:
    
    - In propagate mode (propagate_inf=True): Cycles and hidden loops cause the
      entire result to be marked as infinite (None). Used for detecting algebraic
      loops that affect the entire system.
    - In non-propagate mode (propagate_inf=False): Cycles and hidden loops act as
      termination points (return 0). Used for constructing a DAG from components
      that would otherwise form a cycle.
    
    Parameters
    ----------
    graph_map : dict[Block, set[Block]]
        Adjacency map in the direction of traversal:
        - For upstream traversal: {target : {sources}}
        - For downstream traversal: {source : {targets}}
    node : Block
        Starting block for the traversal.
    node_set : set[Block] | None, optional
        Optional set of blocks to restrict the search to.
        If None, all blocks in the graph_map are considered.
    propagate_inf : bool, default=True
        Controls how cycles and hidden loops are handled:
        - True: Infinite paths propagate and taint the entire result (None)
        - False: Infinite paths simply terminate that branch (0)
    stack : frozenset[Block], optional
        Blocks on the current recursion path (used for cycle detection).
        Internal use only - do not supply manually.
    
    Returns
    -------
    int | None
        - Positive integer: Length of longest finite algebraic path
        - 0: No algebraic path exists (all paths broken by non-algebraic blocks)
        - None: Infinite path length (due to algebraic loops or hidden loops)
    """
    @cache
    def dfs(cur, stk):

        #skip irrelevant nodes -> terminates
        if node_set is not None: 
            if cur not in node_set:
                return 0

        alg_len = len(cur)

        #non-algebraic block -> stops path 
        if alg_len == 0:
            return 0

        #hidden loop block
        if alg_len is None: 

            #taint entire result or terminate
            return None if propagate_inf else 0
            
        #cycle detection
        if cur in stk:

            #taint entire result or terminate
            return None if propagate_inf else 0

        #recurse to neighbours
        best = 0
        nxt_stk = stk | {cur}
        for nbr in graph_map.get(cur, ()):
            
            sub = dfs(nbr, nxt_stk)

            if sub is None:
                #taint entire result or terminate
                return None if propagate_inf else 0

            #update best length
            best = max(best, sub)

        #count this algebraic block
        return best + alg_len

    return dfs(node, stack)


def has_algebraic_path(connection_map, start_block, end_block, node_set=None):
    """Determines if an algebraic path exists between two blocks.
    
    An algebraic path exists if there is a directed path from start_block to 
    end_block consisting entirely of algebraic blocks (blocks with len > 0).
    This function can also detect algebraic self-feedback loops when 
    start_block and end_block are the same.
    
    Parameters
    ----------
    connection_map : dict[Block, set[Block]]
        Adjacency map in downstream (source -> targets) orientation.
    start_block : Block
        Starting block for the path search.
    end_block : Block
        Target block to reach.
    node_set : set[Block] | None, optional
        Optional set of blocks to restrict the search to.
        If None, all blocks in the connection_map are considered.
        
    Returns
    -------
    bool
        True if an algebraic path exists, False otherwise.
        
    Note
    ----
    - Self-feedback loops (when start_block equals end_block) are detected by
      finding a path that returns to the start block after traversing at least
      one other node.
    - Non-algebraic blocks (len == 0) break the path and prevent an algebraic
      connection from being established.
    - The function uses depth-first search with cycle detection to avoid
      infinite recursion.
    """
    # Create a visited set to avoid recomputing nodes
    visited = set()
    
    def dfs(current):
        
        #skip if already visited or not in node_set (if specified)
        if current in visited or (node_set is not None and current not in node_set):
            return False
        
        #mark as visited
        visited.add(current)
        
        #check if we reached the end block
        if current is end_block:
            
            #for self-loops, we need to find a path back to the start
            if start_block is not end_block:
                return True
                
        #non-algebraic blocks break the path
        if len(current) == 0:
            return False
        
        #explore neighbors
        for next_block in connection_map.get(current, ()):

            #self-loops -> True if we find end_block AND it's through a path (not directly)
            if next_block is end_block and start_block is end_block:
                return True
                
            if dfs(next_block):
                return True
                
        return False
    
    #start the search
    return dfs(start_block)


# GRAPH CLASS ==========================================================================

class Graph:
    """Representation of a directed graph, defined by blocks (nodes) 
    and edges (connections).

    Manages graph analysis methods, detect algebraic loops, sort blocks and 
    connections into algebraic levels by depth in directed acyclic graph (DAG). 
    Does the same for algebraic loop blocks and connections (DAG for open loops).

    Parameters
    ----------
    blocks : list[blocks] | None
        blocks / nodes of the graph
    connections : list[Connection] | None
        connections / edges of the graph

    Attributes
    ----------
    has_loops : bool
        flag to indicate if graph has algebraic loops
    _alg_depth : int
        algebraic depth of DAG (number of alg. levels)
    _loop_depth : int
        algebraic depth of broken loop DAG (number of alg. levels in loops)
    _blocks_dag : defaultdict[int, list[Block]]
        algebraic levels of blocks in internal DAG
    _blocks_loop_dag : defaultdict[int, list[Block]]
        algebraic levels of blocks in broken loops DAG
    _connections_dag : defaultdict[int, list[Connection]]
        algebraic levels of connections in internal DAG
    _connections_loop_dag : defaultdict[int, list[Connection]]
        algebraic levels of connections in broken loops DAG
    _upst_blk_blk_map : defaultdict[Block, set[Block]]
        map for upstream connections between blocks
    _dnst_blk_blk_map : defaultdict[Block, set[Block]]
        map for downstream connections between blocks
    _outg_blk_con_map : defaultdict[Block, list[Connection]]
        map for outgoing connections of blocks
    """

    def __init__(self, blocks=None, connections=None):

        self.blocks      = [] if blocks is None else blocks
        self.connections = [] if connections is None else connections

        #loop flag
        self.has_loops = False

        #depths
        self._alg_depth = 0
        self._loop_depth = 0

        #initialize graph orderings
        self._blocks_dag = defaultdict(list)
        self._blocks_loop_dag = defaultdict(list)
        
        self._connections_dag = defaultdict(list)
        self._connections_loop_dag = defaultdict(list)

        #construct mappings for connections between blocks
        self._upst_blk_blk_map = upstream_block_block_map(self.connections)
        self._dnst_blk_blk_map = downstream_block_block_map(self.connections)
        self._outg_blk_con_map = outgoing_block_connection_map(self.connections)

        #assemble dag and loops
        self._assemble()


    def __bool__(self):
        return True


    def __len__(self):
        return len(self.blocks)


    def depth(self):
        return self._alg_depth, self._loop_depth


    def _assemble(self):
        """Constructs two separate DAG orderings of the graph components.
        
        This method performs two key operations:
        
        1. Constructs the main DAG for acyclic portions:
           - Computes algebraic depth for each block using upstream traversal
           - Blocks in algebraic loops are identified (depth = None) and collected
           - Acyclic blocks are placed in _blocks_dag at their calculated depth
           
        2. Constructs a DAG for loop components:
           - Only considers blocks already identified as part of loops
           - Uses non-propagating mode to treat cycles as termination points
           - Organizes loop blocks into levels in _blocks_loop_dag
           
        The result is two separate ordered representations:
        - A standard DAG for acyclic portions (_blocks_dag/_connections_dag)
        - A "broken loop" DAG (_blocks_loop_dag/_connections_loop_dag) that 
          treats loop reentry points as termination points, allowing loops
          to be processed in a deterministic order.
        """

        #collect blocks involved in algebraic loops 
        blocks_loop = []

        #reset flag for loop detection
        self.has_loops = False

        #iterate blocks to calculate their algebraic depths
        for blk in self.blocks:

            #use 'inf' propagating mode to detect loops
            depth = algebraic_depth_dfs(self._upst_blk_blk_map, blk, None, True)

            #None -> upstream alg. loop taints downstream components
            if depth is None:
                blocks_loop.append(blk)
                self.has_loops = True
                
            else:
                #add block to the DAG at its calculated depth
                self._blocks_dag[depth].append(blk)
                self._connections_dag[depth].extend(self._outg_blk_con_map[blk])

        #compute total algebraic depth of DAG
        self._alg_depth = max(self._blocks_dag) + 1 if self._blocks_dag else 0
        
        #build the DAG for the loop blocks by treating cycles as termination points
        for blk in blocks_loop:

            #use non-propagating mode and restrict to only loop blocks
            depth = algebraic_depth_dfs(self._upst_blk_blk_map, blk, blocks_loop, False)

            #add block to the loop DAG at its calculated depth
            self._blocks_loop_dag[depth].append(blk)
            self._connections_loop_dag[depth].extend(self._outg_blk_con_map[blk])

        #finally calculate depth of loop DAG
        self._loop_depth = max(self._blocks_loop_dag) + 1 if self.has_loops else 0


    def outgoing_connections(self, block):
        """Returns the outgoing connections of a block, 
        or, connections that have 'block' as its source
        
        Parameters
        ----------
        block : Block
            block that we want to get the outgoing connections of

        Returns
        -------
        list[Connections]
            connections from the graph that have 'block' as their source
        """
        return self._outg_blk_con_map[block]


    def is_algebraic_path(self, start_block, end_block):
        """Check if two blocks are connected through a 
        purely algebraic path.
        
        Parameters
        ----------
        start_block : Block
            starting block of path
        end_block : Block
            end block of path
        
        Returns
        -------
        bool
            Is there a purely algebraic path between 
            the two blocks? 

        """ 

        #blocks are not part of same graph -> no algebraic path
        if (start_block not in self._dnst_blk_blk_map or 
            end_block not in self._dnst_blk_blk_map):
            return False

        #use depth first search to see if there is a path
        return has_algebraic_path(self._dnst_blk_blk_map, start_block, end_block)


    def dag(self):
        """Generator that yields blocks and connections at each 
        algebraic depth level.
    
        Yields
        ------
        tuple[int, list[Block], list[Connection]]
            blocks and connections at current algebraic depth, 
            together with the depth 'd'
        """
        for d in range(self._alg_depth):
            yield (d, self._blocks_dag[d], self._connections_dag[d])


    def loop(self):
        """Generator that yields blocks and connections that are part of 
        algebraic loops. 

        Formatted as a DAG, that represents a broken loop with entry points.

        Yields
        ------
        tuple[int, list[Block], list[Connection]]
            blocks and connections at current algebraic depth of 
            broken loop, together with the depth 'd'
        """
        for d in range(self._loop_depth):
            yield (d, self._blocks_loop_dag[d], self._connections_loop_dag[d])