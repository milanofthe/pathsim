########################################################################################
##
##                            FUNCTIONS FOR GRAPH ANALYSIS
##                                  (utils/graph.py)
##
##                                 Milan Rother 2025
##
########################################################################################

# IMPORTS ==============================================================================

from collections import defaultdict
from functools import cache


# PATH ESTIMATION ======================================================================

def downstream_connection_map(connections):
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


def upstream_connection_map(connections):
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
    """
    Return a triple: (alg_len, is_hidden_loop, is_algebraic)

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
    """
    Longest algebraic path length ending at `starting_block`, walking
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
    """
    Longest algebraic path length starting from `starting_block`, walking
    **downstream** (source → targets).

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