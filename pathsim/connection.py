#########################################################################################
##
##                            CONNECTION CLASS (connection.py)
##
##              This module implements the 'Connection' class that transfers
##                data between the blocks and their input/output channels
##
##                                  Milan Rother 2023/24
##
#########################################################################################

# IMPORTS ===============================================================================

#no dependencies


# CLASSES ===============================================================================

class Connection:

    """
    Class to handle input-output relations of blocks by connecting them (directed graph) 
    and transfering data from the output port of the source block to the input port of 
    the target block.

    The default ports for connection are (0) -> (0), since these are the default inputs 
    that are used in the SISO blocks.

    EXAMPLE:

        Lets assume we have two generic blocks 

            B1 = Block...
            B2 = Block...

        that we want to connect. We initialize a 'Connection' with the blocks directly 
        as the arguments if we want to connect the default ports (0) -> (0) 

            C = Connection(B1, B2)

        which is a connection from block 'B1' to 'B2'. If we want to explicitly declare 
        the input and output ports we can do that by giving tuples (lists also work) as 
        the arguments
     
            C = Connection((B1, 0), (B2, 0))

        which is exactly the default port setup. Connecting output port (1) of 'B1' to 
        the default input port (0) of 'B2' do

            C = Connection((B1, 1), (B2, 0))

        or just

            C = Connection((B1, 1), B2).

        The 'Connection' class also supports multiple targets for a single source. 
        This is specified by just adding more blocks with their respective ports into 
        the constructor like this:

            C = Connection(B1, (B2, 0), (B2, 1), B3, ...)

        The port definitions follow the same structure as for single target connections.

        'self'-connections also work without a problem. This is useful for modeling direct 
        feedback of a block to itself.
        
        The port specification can be simplified (quality of life) by using the __getitem__ 
        method that is implemented in the base 'Block' class. It returns the tuple of block
        and port pair that is used for the port specification in the 'Connection' 
        initialization. For example the following initializations are equivalent:

            Connection(B1[1], B2[3]) <=> Connection((B1, 1), (B2, 3))

    INPUTS: 
        source  : (tuple ('Block', int) OR 'Block') source block and optional source output port
        targets : (tuples of ('Block' int) OR multiple 'Block's) target blocks and optional target input ports
    """

    def __init__(self, source, *targets):
        
        #assign source block and port
        self.source = source if isinstance(source, (list, tuple)) else (source, 0)

        #assign target blocks and ports
        self.targets = [trg if isinstance(trg, (list, tuple)) else (trg, 0) for trg in targets]


    def __str__(self):
        src, prt = self.source
        return f"Connection from ({src}, {prt}) to " + ", ".join([ f"({trg}, {prt})" for trg, prt in self.targets])


    def overwrites(self, other):
        """
        Check if the connection 'self' overwrites the target port
        of connection 'other' and return 'True' if so.

        INPUTS:
            other : ('Connection' instance) other connection to check 
        """

        #catch self checking
        if self == other:
            return False

        #iterate all targets
        for trg in self.targets:
            #check if there is target and port overlap
            if trg in other.targets:
                return True

        return False


    def update(self):
        """
        Transfers data from the source block output port 
        to the target block input port.
        """
        val = self.source[0].get(self.source[1])
        for trg, prt in self.targets:
            trg.set(prt, val)