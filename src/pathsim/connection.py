#########################################################################################
##
##                                   CONNECTION CLASS 
##                                    (connection.py)
##
##              This module implements the 'Connection' class that transfers
##                data between the blocks and their input/output channels
##
##                               Milan Rother 2023/24/2025
##
#########################################################################################

# IMPORTS ===============================================================================

import json

from .utils.portreference import PortReference

from .optim.anderson import Anderson


# CLASSES ===============================================================================

class Connection:
    """Class to handle input-output relations of blocks by connecting them (directed graph) 
    and transfering data from the output port of the source block to the input port of 
    the target block.

    The default ports for connection are (0) -> (0), since these are the default inputs 
    that are used in the SISO blocks.

    Examples
    --------
    Lets assume we have some generic blocks 

    .. code-block:: python
    
        from pathsim.blocks._block import Block

        B1 = Block()
        B2 = Block()
        B3 = Block()


    that we want to connect. We initialize a 'Connection' with the blocks directly 
    as the arguments if we want to connect the default ports (0) -> (0) 
    
    .. code-block:: python

        from pathsim import Connection

        C = Connection(B1, B2)


    which is a connection from block 'B1' to 'B2'. If we want to explicitly declare 
    the input and output ports we can do that by utilizing the '__getitem__' method
    of the blocks

    .. code-block:: python
 
        C = Connection(B1[0], B2[0])


    which is exactly the default port setup. Connecting output port (1) of 'B1' to 
    the default input port (0) of 'B2' do

    .. code-block:: python

        C = Connection(B1[1], B2[0])
        

    or just
    
    .. code-block:: python

        C = Connection(B1[1], B2).


    The 'Connection' class also supports multiple targets for a single source. 
    This is specified by just adding more blocks with their respective ports into 
    the constructor like this:
    
    .. code-block:: python

        C = Connection(B1, B2[0], B2[1], B3)


    The port definitions follow the same structure as for single target connections.

    'self'-connections also work without a problem. This is useful for modeling direct 
    feedback of a block to itself.
    
    Port definitions support slicing. This enables direct MIMO connections. For example 
    connecting ports 0, 1, 2 of 'B1' to ports 1, 2, 3 of 'B2' works like this:

    .. code-block:: python

        C = Connection(B1[0:2], B2[1:3])

    
    Port definitions also support lists and tuples of 'int'. For example the slice 
    above is identical to this:
    
    .. code-block:: python

        C = Connection(B1[0, 1], B2[1, 2])


    Or to be more programmatic about it, like this:

    .. code-block:: python
    
        prts_1 = [0, 1]
        prts_2 = [1, 2]

        C = Connection(B1[prts_1], B2[prts_2])


    Parameters
    ----------
    source : PortReference, Block
        source block and optional source output port
    targets : tuple[PortReference], tuple[Block]
        target blocks and optional target input ports


    Attributes
    ----------
    _active : bool
        flag to set 'Connection' as active or inactive
    values : array
        values to transmit, relevant for fixed-point accelerator
    accelerator : None, Anderson
        internal fixed-point accelerator for algebraic loops
    """

    __slots__ = ["source", "targets", "_active", "values", "accelerator"]


    def __init__(self, source, *targets):
        
        #assign source block and port
        self.source = source if isinstance(source, PortReference) else PortReference(source)

        #assign target blocks and ports
        self.targets = [trg if isinstance(trg, PortReference) else PortReference(trg) for trg in targets]

        #flag to set connection active
        self._active = True

        #values to transmit as history
        self.values = None

        #internal fixed-point accelerator
        self.accelerator = None
        
        #validate port dimensions at connection creation
        self._validate_dimensions()        


    def __str__(self):
        """String representation of the connection using the 
        'to_dict' method with readable json formatting
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=False)


    def __bool__(self):
        return self._active


    def __contains__(self, other):
        """Check if block is part of connection

        Paramters
        ---------
        other : Block
            block to check if its part of the connection

        Returns
        -------
        bool
            is other part of connecion?
        """
        if isinstance(other, Block): 
            return other in self.get_blocks()
        return False


    def _validate_dimensions(self):
        """Check the dimensions of the source and target ports, 
        if they dont match, raises an exception.
        """
        n_src = len(self.source)
        for trg in self.targets:
            if len(trg) != n_src:
                raise ValueError(f"Source and target have different number of ports!")


    def get_blocks(self):
        """Returns all the unique internal source and target blocks 
        of the connection instance

        Returns
        -------
        list[Block]
            internal unique blocks of the connection
        """
        blocks = [self.source.block]
        for trg in self.targets:
            if trg.block not in blocks:
                blocks.append(trg.block)
        return blocks


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

        #iterate all target permutations
        for trg in self.targets:
            for otrg in other.targets:

                #check if same target block
                if trg.block is otrg.block:

                    #check if there is port overlap
                    for prt in trg.ports:
                        if prt in otrg.ports: 
                            return True

        return False 


    def to_dict(self):
        """Convert connection to dictionary representation for serialization"""
        return {
            "id": id(self),
            "source": self.source.to_dict(),
            "targets": [trg.to_dict() for trg in self.targets]
        }


    def update(self):
        """Transfers data from the source block output port 
        to the target block input port.
        """
        for trg in self.targets:
            self.source.to(trg)


    def step(self):
        """Step the internal fixed-point accelerator forward by one iteration.

        If no previous values are available (prev_values is None), falls back to 
        the 'update' method, which is essentially equivalent to a standard 
        fixed-point update. 
    
        Returns
        -------
        res : float
            fixed point residual for convergence control
        """

        #get source values and previous values
        self.values, prev_values = self.source.get_outputs(), self.values

        #initialize fixed point accelerator if not already available
        if not self.accelerator:
            self.accelerator = Anderson()

        #no previous value -> fallback to update method
        if prev_values is None:
            self.update()
            return 1.0

        #update fixed-point accelerator
        self.values, res = self.accelerator.step(prev_values, self.values)

        #transmit new values to all targets
        for trg in self.targets:
            trg.set_inputs(self.values)

        #return the fixed-point residual
        return res


    def reset(self):
        """Reset the internal fixed point accelerator which is used 
        to resolve algebraic loops and the internal values"""
        self.values = None
        if self.accelerator: 
            self.accelerator.reset()


class Duplex(Connection):
    """Extension of the 'Connection' class, that defines bidirectional 
    connections between two blocks by grouping together the inputs and 
    outputs of the blocks into an IO-pair.
    """

    __slots__ = ["source", "target", "targets", "_active"]


    def __init__(self, source, target):
        
        self.source = source if isinstance(source, PortReference) else PortReference(source)
        self.target = target if isinstance(target, PortReference) else PortReference(target)
        
        #this is required for path length estimation
        self.targets = [self.target, self.source]

        #flag to set connection active
        self._active = True
        

    def to_dict(self):
        """Convert duplex to dictionary representation for serialization"""
        return {
            "id": id(self),
            "source": self.source.to_dict(),
            "target": self.target.to_dict()
        }


    def update(self):
        """Transfers data between the two target blocks 
        and ports bidirectionally.
        """

        #bidirectional data transfer
        self.target.to(self.source)
        self.source.to(self.target)
