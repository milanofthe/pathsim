########################################################################################
##
##                            FUNCTIONS FOR GRAPH ANALYSIS
##                                  (utils/graph.py)
##
########################################################################################

# IMPORTS ==============================================================================

from collections import defaultdict, deque, namedtuple
from functools import lru_cache


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
        outgoing connections of block (deterministic order)
    """
    block_connection_map = defaultdict(list)

    # Build map grouping connections by source block.
    for con in connections:
        src_blk = con.source.block
        block_connection_map[src_blk].append(con)

    # Sort connections per source deterministically by (src_key, targets_keys...)
    def _conn_key(c):
        src_k = id(c.source.block)
        # produce stable list of target keys
        tgt_keys = tuple(sorted((id(t.block) for t in c.targets)))
        return (src_k, tgt_keys)

    for src in list(block_connection_map.keys()):
        block_connection_map[src].sort(key=_conn_key)

    return defaultdict(list, block_connection_map)


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

    # Convert to defaultdict of frozensets for stable internal use
    return defaultdict(set, {k: set(v) for k, v in connection_map.items()})


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

    return defaultdict(set, {k: set(v) for k, v in connection_map.items()})


def algebraic_depth_dfs(
    graph_map,
    node,
    node_set=None,
    propagate_inf=True,
    stack=None
    ):
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
    stack : tuple[Block, ...], optional
        Blocks on the current recursion path (used for cycle detection).
        Internal use only - do not supply manually.

    Returns
    -------
    int | None
        - Positive integer: Length of longest finite algebraic path
        - 0: No algebraic path exists (all paths broken by non-algebraic blocks)
        - None: Infinite path length (due to algebraic loops or hidden loops)
    """
    if stack is None:
        stack = tuple()

    # Use lru_cache for memoization. The cache key will be based on the node and
    # propagate_inf flag; node_set restriction is encoded by checking membership.
    @lru_cache(maxsize=None)
    def _dfs_cached(cur):
        # The stack is captured from the caller frame via closure. To maintain
        # a proper cycle detection, we implement a small explicit recursion wrapper.

        return _dfs(cur, stack)

    def _dfs(cur, stk):
        # Skip irrelevant nodes -> terminates
        if node_set is not None and cur not in node_set:
            return 0

        # Algebraic length of current block. len(block) yields
        # - 0 : non-algebraic (terminates)
        # - positive int : algebraic order/count
        # - None : hidden loop block
        alg_len = len(cur)

        # non-algebraic block -> stops path
        if alg_len == 0:
            return 0

        # hidden loop block
        if alg_len is None:
            return None if propagate_inf else 0

        # cycle detection
        if cur in stk:
            return None if propagate_inf else 0

        # Recurse to neighbours. To be deterministic, iterate neighbours in sorted order.
        neighbors = graph_map.get(cur, ())
        # neighbors may be a set, convert to list sorted by stable key
        nbrs_sorted = sorted(neighbors, key=id)

        best = 0
        next_stk = stk + (cur,)

        for nbr in nbrs_sorted:
            # enforce node_set restriction
            if node_set is not None and nbr not in node_set:
                continue

            # Use cached recursion by node identity (since stack varies we can't
            # cache on stack; however caching result for a node alone is safe
            # because propagate_inf and node_set are fixed for the whole outer call).
            # We implement manual caching using a dict to avoid reconstructing function.
            sub = _dfs(nbr, next_stk)

            if sub is None:
                # taint entire result or terminate
                return None if propagate_inf else 0

            # update best length
            if sub > best:
                best = sub

        # count this algebraic block
        return best + alg_len

    # Call the dfs with top-level stack
    return _dfs(node, stack)


def has_algebraic_path(
    connection_map,
    start_block,
    end_block,
    node_set=None
    ):
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
      infinite recursion. Traversal order is deterministic.
    """
    # quick existence checks
    if start_block not in connection_map and start_block not in (connection_map.keys()):
        # Might still be reachable if start_block has no outgoing but is end_block
        # In that case, only a self-loop qualifies (len>0 and path length > 0)
        if start_block is not end_block:
            return False

    visited = set()

    # Iterative DFS stack: list of (node, iterator over sorted neighbors, depth_from_start)
    # depth_from_start helps detect trivial self-loop (path must traverse at least one other node)
    stack = [(start_block, iter(sorted(connection_map.get(start_block, ()), key=id)), 0)]

    while stack:
        node, nbr_iter, depth = stack[-1]
        # mark visited when popped? We mark when pushed to prevent revisiting within same path
        if node not in visited:
            visited.add(node)

        try:
            nbr = next(nbr_iter)
        except StopIteration:
            stack.pop()
            continue

        # skip if node_set restricts it
        if node_set is not None and nbr not in node_set:
            continue

        # If we've reached the end_block
        if nbr is end_block:
            # If it's not a self-loop case (start != end) this is success
            if start_block is not end_block:
                # ensure nbr is algebraic
                if len(nbr) != 0:
                    return True
                else:
                    # target is non-algebraic, cannot form algebraic path
                    continue
            else:
                # start==end: must traverse at least one other node before returning
                if depth + 1 >= 1 and len(nbr) != 0:
                    return True
                else:
                    # treat it as a neighbor but not success yet
                    pass

        # skip non-algebraic neighbors
        if len(nbr) == 0:
            continue

        # If not visited, push onto stack with its neighbor iterator
        if nbr not in visited:
            stack.append((nbr, iter(sorted(connection_map.get(nbr, ()), key=id)), depth + 1))

    return False


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

        self.blocks = [] if blocks is None else list(blocks)
        self.connections = [] if connections is None else list(connections)

        # loop flag
        self.has_loops = False

        # depths
        self._alg_depth = 0
        self._loop_depth = 0

        # initialize graph orderings
        self._blocks_dag = defaultdict(list)
        self._blocks_loop_dag = defaultdict(list)

        self._connections_dag = defaultdict(list)
        self._connections_loop_dag = defaultdict(list)
        self._loop_closing_connections = []

        # construct mappings for connections between blocks (internally sets)
        self._upst_blk_blk_map = upstream_block_block_map(self.connections)
        self._dnst_blk_blk_map = downstream_block_block_map(self.connections)
        # outgoing connection map stays list but sorted per source to be deterministic
        self._outg_blk_con_map = outgoing_block_connection_map(self.connections)

        # assemble dag and loops
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
           - Identifies strongly connected components (SCCs) to find separate loops
           - For each SCC, finds proper entry points
           - Builds a level structure for each loop using BFS from entry points
           - Organizes all loop blocks into levels in _blocks_loop_dag

        The result is two separate ordered representations:
        - A standard DAG for acyclic portions (_blocks_dag/_connections_dag)
        - A "broken loop" DAG (_blocks_loop_dag/_connections_loop_dag) with
          proper entry points and topological ordering for all loops.
        """
        # reset structures
        self._blocks_dag.clear()
        self._connections_dag.clear()
        self._blocks_loop_dag.clear()
        self._connections_loop_dag.clear()
        self._loop_closing_connections.clear()
        self.has_loops = False

        # collect blocks involved in algebraic loops
        blocks_loop = set()

        # iterate blocks to calculate their algebraic depths deterministically:
        # sort blocks by stable key before iterating to avoid input-order dependence
        sorted_blocks = sorted(self.blocks, key=id)

        for blk in sorted_blocks:
            depth = algebraic_depth_dfs(self._upst_blk_blk_map, blk, None, True)

            if depth is None:
                blocks_loop.add(blk)
                self.has_loops = True
            else:
                # add block to the DAG at its calculated depth
                self._blocks_dag[depth].append(blk)
                # append outgoing connections deterministically (already sorted in map)
                for con in self._outg_blk_con_map.get(blk, ()):
                    self._connections_dag[depth].append(con)

        # compute total algebraic depth of DAG
        self._alg_depth = (max(self._blocks_dag) + 1) if self._blocks_dag else 0

        # if no loops, done
        if not self.has_loops:
            self._loop_depth = 0
            return

        # find strongly connected components among blocks in loops
        sccs = self._find_strongly_connected_components(sorted(list(blocks_loop), key=id))

        # track global depth counter for all loop blocks
        current_depth = 0

        # Process each strongly connected component separately, deterministically
        for scc in sccs:
            scc_set = set(scc)

            # find entry points for this SCC deterministically
            entry_points = []
            for blk in sorted(scc, key=id):
                pred = self._upst_blk_blk_map.get(blk, set())
                scc_pred = pred.intersection(scc_set)

                # A block is an entry point if:
                # 1. It has no predecessors within this SCC, or
                # 2. It has at least one predecessor outside this SCC
                if not scc_pred or len(pred) > len(scc_pred):
                    entry_points.append(blk)

            # If no natural entry points, choose first block as artificial entry
            if not entry_points:
                entry_points = [sorted(scc, key=id)[0]]

            # BFS from entry points for this SCC with deterministic neighbor ordering
            visited = set()
            queue = deque([(ep, 0) for ep in sorted(entry_points, key=id)])
            max_local_depth = 0
            local_depths: Dict[Any, int] = {}

            while queue:
                blk, local_depth = queue.popleft()
                if blk in visited:
                    # only keep smallest depth already discovered
                    if local_depth < local_depths.get(blk, float("inf")):
                        local_depths[blk] = local_depth
                        max_local_depth = max(max_local_depth, local_depth)
                    continue

                visited.add(blk)
                local_depths[blk] = local_depth
                max_local_depth = max(max_local_depth, local_depth)

                # Enqueue downstream neighbors that are in this SCC in deterministic order
                for next_blk in sorted(self._dnst_blk_blk_map.get(blk, ()), key=id):
                    if next_blk in scc_set and next_blk not in visited:
                        queue.append((next_blk, local_depth + 1))

            # Second pass: assign global depths and classify connections
            for blk in sorted(scc, key=id):
                global_depth = current_depth + local_depths.get(blk, 0)
                self._blocks_loop_dag[global_depth].append(blk)

                # for each outgoing connection from this block determine if it closes the loop
                for con in self._outg_blk_con_map.get(blk, ()):
                    is_loop_closing = False
                    for target in con.targets:
                        target_blk = target.block
                        if target_blk in scc_set:
                            # back edge if target depth <= blk depth
                            if local_depths.get(target_blk, 0) <= local_depths.get(blk, 0):
                                # loop closing connection
                                self._loop_closing_connections.append(con)
                                is_loop_closing = True
                                break
                    if not is_loop_closing:
                        self._connections_loop_dag[global_depth].append(con)

            # update global depth counter for the next SCC
            current_depth += max_local_depth + 1

        # compute depth of loop DAG
        self._loop_depth = (max(self._blocks_loop_dag) + 1) if self._blocks_loop_dag else 0


    def _find_strongly_connected_components(self, blocks):
        """Finds strongly connected components (SCCs) in the subgraph
        defined by blocks.

        Uses Tarjan's algorithm to identify separate loop structures.

        Parameters
        ----------
        blocks : list[Block]
            Blocks to consider for SCC analysis

        Returns
        -------
        list[list[Block]]
            List of SCCs, where each SCC is a list of blocks
        """
        block_set = set(blocks)
        index_counter = [0]
        index = {}
        lowlink = {}
        onstack = set()
        stack = []
        result = []

        # deterministic neighbor ordering helper
        def _successors(node):
            return sorted((n for n in self._dnst_blk_blk_map.get(node, ()) if n in block_set), key=id)

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            onstack.add(node)

            for succ in _successors(node):
                if succ not in index:
                    strongconnect(succ)
                    lowlink[node] = min(lowlink[node], lowlink[succ])
                elif succ in onstack:
                    lowlink[node] = min(lowlink[node], index[succ])

            # If node is a root node, pop the stack and generate an SCC
            if lowlink[node] == index[node]:
                scc = []
                while True:
                    w = stack.pop()
                    onstack.remove(w)
                    scc.append(w)
                    if w == node:
                        break
                # Only keep SCCs that are actual cycles: size>1 or self-loop
                if len(scc) > 1 or any(node in self._dnst_blk_blk_map.get(node, ()) for node in scc):
                    # sort scc deterministically before appending
                    result.append(sorted(scc, key=id))

        # Process nodes in deterministic order
        for node in sorted(blocks, key=id):
            if node not in index:
                strongconnect(node)

        return result

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
        # blocks not present in downstream map -> no algebraic path
        if start_block not in self._dnst_blk_blk_map and end_block not in self._dnst_blk_blk_map:
            return False

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

    def loop_closing_connections(self):
        """Returns the connections that close algebraic loops

        Returns
        -------
        list[Connection]
            Connections that close the algebraic loops from the broke loop DAG
        """
        return self._loop_closing_connections
