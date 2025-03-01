#########################################################################################
##
##                             CONNECTION CLASS (connection.py)
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
    """Class to handle input-output relations of blocks by connecting them (directed graph) 
    and transfering data from the output port of the source block to the input port of 
    the target block.

    The default ports for connection are (0) -> (0), since these are the default inputs 
    that are used in the SISO blocks.

    Example
    -------
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

    Parameters
    ----------
    source : tuple[Block, int], Block
        source block and optional source output port
    targets : tuple[tuple[Block, int]], tuple[Block]
        target blocks and optional target input ports
    """

    def __init__(self, source, *targets):
        
        #assign source block and port
        self.source = source if isinstance(source, (list, tuple)) else (source, 0)

        #assign target blocks and ports
        self.targets = [trg if isinstance(trg, (list, tuple)) else (trg, 0) for trg in targets]

        #flag to set connection active
        self._active = True


    def __str__(self):
        src, prt = self.source
        return f"Connection ({src}, {prt}) -> " + ", ".join([ f"({trg}, {prt})" for trg, prt in self.targets])


    def __bool__(self):
        return self._active


    def on(self):
        self._active = True


    def off(self):
        self._active = False


    def overwrites(self, other):
        """Check if the connection 'self' overwrites the target port of 
        connection 'other' and return 'True' if so.
    
        Parameters
        ----------
        other : Connection
            other connection to check 

        Returns
        -------
        overwrites : bool
            True if port is overwritten, False otherwise

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


    def to_dict(self):
        """Convert connection to dictionary representation for serialization"""
        return {
            "id": id(self),
            "source": {
                "block": id(self.source[0]),
                "port": self.source[1]
            },
            "targets": [
                {"block": id(trg[0]), "port": trg[1]} 
                for trg in self.targets
            ]
        }


    def update(self):
        """Transfers data from the source block output port 
        to the target block input port.
        """
        val = self.source[0].get(self.source[1])
        for trg, prt in self.targets:
            trg.set(prt, val)


class Duplex(Connection):
    """Extension of the 'Connection' class, that defines bidirectional 
    connections between two blocks by grouping together the inputs and 
    outputs of the blocks into an IO-pair.
    """

    def __init__(self, source, target):
        
        self.source = source if isinstance(source, (list, tuple)) else (source, 0)
        self.target = target if isinstance(target, (list, tuple)) else (target, 0)
        
        #for path length estimation
        self.targets = [self.target]

        #flag to set connection active
        self._active = True
        

    def __str__(self):
        return f"Duplex {self.source} <-> {self.target}" 


    def update(self):
        """Transfers data between the two target blocks 
        and ports bidirectionally.
        """

        #unpack the two targets
        (trg1, prt1) = self.target
        (trg2, prt2) = self.source

        #bidirectional data transfer
        trg1.set(prt1, trg2.get(prt2))
        trg2.set(prt2, trg1.get(prt1))