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


def _block_meta(block):
    """Return a triple: (alg_len, is_hidden_loop, is_algebraic)

    * `len(block) is None` -> (0,  True,  False)
    * `len(block) == 0`    -> (0,  False, False)
    * `len(block) > 0`     -> (len, False, True)
    """
    raw = len(block)
    if raw is None:
        return 0, True, False
    return raw, False, raw > 0


def _dfs_generic(graph_map, node, propagate_inf, start, stack=frozenset()):
    """Internal DFS that implements all path-length semantics.

    Parameters
    ----------
    graph_map : dict[Block, set[Block]]
        Neighbour map *in the traversal direction*:
        * upstream call  -> {target : {sources}}
        * downstream call -> {source : {targets}}

    node : Block
        Current block in the recursion.

    propagate_inf : bool
        *True*  ->  an 'infinite' branch (algebraic loop or hidden-loop block)
        propagates upstream to taint the **entire** result (`None`).
        (*Used by upstream_path_length_dfs*.)

        *False* ->  an infinite branch is clipped locally; the overall
        result is only `None` if the **start** block itself lies in/with
        an algebraic loop or is a hidden-loop block.
        (*Used by downstream_path_length_dfs*.)

    start : Block
        Block for which the public API was invoked; needed to recognise
        whether a detected cycle involves the start node.

    stack : frozenset[Block], optional
        Blocks on the current recursion path (immutable for hashability).
        Do **not** supply manually; the first call leaves this argument
        at its default.

    Returns
    -------
    int | None
        * Positive integer – longest finite algebraic path length.
        * 0 – no algebraic blocks reachable before a path-breaking
          non-algebraic block.
        * None – result is “infinite” under the rules described above.
    """
    @cache
    def dfs(cur, stk):
        alg_len, hidden_loop, is_alg = _block_meta(cur)

        #hidden loop block
        if hidden_loop:
            return None if (propagate_inf or cur is start) else 0

        #cycle detection
        if cur in stk:
            #non-algebraic blocks stop the walk
            return None if (propagate_inf or cur is start) else 0

        #non-algebraic block -> stops path 
        if alg_len == 0:
            return 0

        #recurse to neighbours
        best = 0
        nxt_stk = stk | {cur}
        for nbr in graph_map.get(cur, ()):
            
            sub = dfs(nbr, nxt_stk)

            if sub is None:

                if propagate_inf:
                    #taint entire result
                    return None           

                #downstream mode –> ignore this branch
                continue

            #update best length
            best = max(best, sub)

        #count this algebraic block
        return best + alg_len              

    return dfs(node, stack)


def upstream_path_length_dfs(connection_map, starting_block):
    """Longest algebraic path length ending at `starting_block`, walking
    **upstream** (targets -> sources).

    * Stops – and finishes that branch – as soon as a *non-algebraic* block
      (`len==0`) is reached.

    * If any upstream branch hits an *algebraic loop* **or** a *hidden-loop*
      block (`len is None`), the path length is considered **infinite** and
      the function returns **None**.

    * Otherwise returns the maximum finite algebraic length (>=0).
    """
    return _dfs_generic(
        connection_map,
        starting_block,
        propagate_inf=True,   # upstream -> infinite branches taint whole result
        start=starting_block
    )


def downstream_path_length_dfs(connection_map, starting_block):
    """Longest algebraic path length starting from `starting_block`, 
    walking **downstream** (source -> targets).

    * Traversal *immediately* stops on a non-algebraic block, a hidden-loop
      block, or a loop farther downstream; those branches contribute 0.

    * The result is **None** only if the starting block itself is part of an
      algebraic loop or is a hidden-loop block.  Branch-local loops do *not*
      taint the answer.

    * Otherwise returns the maximum finite algebraic length (≥0).
    """
    return _dfs_generic(
        connection_map,
        starting_block,
        propagate_inf=False,  # downstream -> infinite branches local
        start=starting_block
    )


def distance_path_length_dfs(connection_map, start_block, end_block):
    """Return the length of the *longest purely-algebraic* directed path 
    from `src_block` to `dst_block` following the `connection_map` 
    orientation (source -> targets).

    Parameters
    ----------
    connection_map : dict[Block, set[Block]]
        Adjacency map in downstream (source -> targets) orientation.
    start_block, end_block : Block
        Start and end nodes of interest.  They may be identical; a zero-length
        self-path is then possible.

    Returns
    -------
    int | None
        * 0      – No algebraic path exists (either no path at all, or every
                   path is broken by at least one non-algebraic block
                   ``len(b)==0``).
        * >0     – Length of the longest purely-algebraic path (sum of
                   ``len(b)`` for all blocks on that path).
        * None   – At least one candidate path runs into
                     * an **algebraic loop** (cycle containing a
                       ``len(b)>0`` block), **or**
                     * a **hidden-loop block** where ``len(b) is None``.
                   Because such a loop makes the algebraic influence
                   'infinite', we propagate the value `None`.
    """
    @cache
    def dfs(node, stack=frozenset()):
        n_len = len(node)
        
        #hidden loop block -> abort with None
        if n_len is None:
            return None

        #block is non-algebraic -> breaks path
        if n_len == 0:
            return 0

        #reached destination after leaving it once -> base length (include end if algebraic)
        if node is end_block and stack:
            return n_len

        #cycle detection
        if node in stack:
            
            #least one algebraic block? -> algebraic cycle  
            cycle_alg = any(len(b) and len(b) > 0 for b in stack | {node})
            return None if cycle_alg else 0

        #explore targets
        best = 0
        for nxt in connection_map.get(node, ()):

            #skip non alg targets


            sub = dfs(nxt, stack | {node})

            #loop upstream of end -> propagate
            if sub is None:
                return None          
            
            best = max(best, sub)
        
        #if all branches broken -> 0
        return (best + n_len) 

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


    def _build_loop_depths(self, blocks_loop):
        """Populate the defaultdicts that order **only the blocks/connections
        which belong to algebraic loops** by an internal “depth”.

        A purely internal, breadth-first layering is used:

        1.  **Entry set**  (depth 0) = every block in the loop sub-graph
            that either
              * has an upstream predecessor outside the loop, **or**
              * has no predecessors at all
            If the SCC is completely self-contained (no entry), pick one
            arbitrary seed so the loop still gets an ordering.

        2.  Perform a BFS that walks *only edges whose target is also in the
            loop.*  Every time we traverse an algebraic edge we enqueue the
            target with depth = parent + 1.

        The result is stored in:

        .. code-block::

            self._blocks_loop_dag[depth]       ->  list[Block]
            self._connections_loop_dag[depth]  ->  list[Connection]

        Depth values are purely relative but let the solver update loop
        blocks in Gauss–Seidel order (shallow -> deep) if desired.
        """

        #safety, nothing to do
        if not blocks_loop: return

        loop_set = set(blocks_loop)
        depth_of  = {}

        #collect entry nodes
        for blk in loop_set:
            preds = self._upst_blk_blk_map[blk]
            if not preds or any(p not in loop_set for p in preds):
                depth_of[blk] = 0
                self._blocks_loop_dag[0].append(blk)

        #self-contained SCC -> seed with first block
        if not depth_of:
            seed = blocks_loop[0]
            depth_of[seed] = 0
            self._blocks_loop_dag[0].append(seed)

        #BFS level-by-level inside the loop
        q = deque(self._blocks_loop_dag[0])

        while q:
            cur = q.popleft()
            cur_d = depth_of[cur]

            #add outgoing connections at this depth (for convenience)
            self._connections_loop_dag[cur_d].extend(self._outg_blk_con_map[cur])

            for nxt in self._dnst_blk_blk_map[cur]:
                
                #leaves the loop -> ignore
                if nxt not in loop_set: continue

                #already visited
                if nxt in depth_of: continue
                
                depth_of[nxt] = cur_d + 1
                self._blocks_loop_dag[cur_d + 1].append(nxt)
                
                q.append(nxt)

        #finally calculate depth of loop DAG
        self._loop_depth = max(self._blocks_loop_dag) + 1


    def _assemble(self):
        """Assemble components, ordered by their algebraic depth in 
        the DAG.

        Perform upstream depth first search on the DAG to assign 
        each block their number of consecutive algebraic dependencies 
        (algebraic depth). This is also used to assign the outgoing 
        connections similarly.
        """

        #collect tainted blocks 
        blocks_loop = []

        #flag for loop detection
        self.has_loops = False

        #iterate blocks to calculate their algebraic depths
        for blk in self.blocks:
            depth = upstream_path_length_dfs(self._upst_blk_blk_map, blk) 

            #None -> alg. loop upstream taints downstream components
            if depth is None:
                blocks_loop.append(blk)
                self.has_loops = True
                
            else:
                self._blocks_dag[depth].append(blk)
                self._connections_dag[depth].extend(self._outg_blk_con_map[blk])

        #compute total algebraic depth of DAG
        self._alg_depth = max(self._blocks_dag) + 1 if self._blocks_dag else 0
        
        #build the DAG for the broken loops
        self._build_loop_depths(blocks_loop)


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


    def distance(self, start_block, end_block):
        """Compute the algebraic distance / path length between two blocks 
        in the downstream arangement ('start_block' -> 'end_block') in the 
        graph.
        """

        #blocks are not part of same graph -> no algebraic path
        if (start_block not in self._dnst_blk_blk_map or 
            end_block not in self._dnst_blk_blk_map):
            return 0

        #use depth first search
        return distance_path_length_dfs(self._dnst_blk_blk_map, start_block, end_block)


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