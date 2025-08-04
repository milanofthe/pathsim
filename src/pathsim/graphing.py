from graphviz import Digraph

def graph(blocks, connections, file_name='block_diagram'):
    """Generate a block diagram from blocks and connections."""
    dot = Digraph(comment="", format='png')
    dot.attr(rankdir='LR')  # Left-to-right layout
    dot.attr(splines='ortho')  # Use orthogonal edges

    # Add nodes for each block
    for block in blocks:
        dot.node(str(id(block)), block.__class__.__name__, shape='rectangle')

    dot_count = 0
    for conn in connections:
        src = str(id(conn.source.block))
        if len(conn.targets) > 1:
            # Create a black dot node for splitting
            dot_node = f'dot{dot_count}'
            dot.node(dot_node, '', shape='circle', width='0.1', fixedsize='true', style='filled', color='black')
            # Line (no arrow) from source to dot
            dot.edge(src, dot_node, arrowhead='none')
            # Normal arrows from dot to each target
            for target in conn.targets:
                dst = str(id(target.block))
                dot.edge(dot_node, dst)
            dot_count += 1
        else:
            for target in conn.targets:
                dst = str(id(target.block))
                dot.edge(src, dst)

    dot.render(f'{file_name}', view=True)