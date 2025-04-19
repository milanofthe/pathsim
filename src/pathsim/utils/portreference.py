#########################################################################################
##
##                                PORT REFERENCE CLASS 
##                              (utils/portreference.py)
##                              
##                                 Milan Rother 2025
##
#########################################################################################

# IMPORTS ===============================================================================

from itertools import cycle

from .blocks._block import Block


# CLASS =================================================================================

class PortReference:
    """This is a container class that holds a reference to a block 
    and a list of ports.

    Note
    ----
    The default port, when no ports are defined in the arguments is `0`.

    Parameters
    ----------
    block : Block
        internal block reference
    ports : list[int]
        list of port indices
    """

    def __init__(self, block=None, ports=None):

        #input type validation
        if not isinstance(block, Block):
            raise ValueError(f"'block' must be type 'Block' or subclass but is '{type(block)}'!")
        if not isinstance(ports, list):            
            raise ValueError(f"'ports' must be 'list[int]' but is '{type(ports)}'!")

        #unique ports
        if len(ports) != len(set(ports)):
            raise ValueError("'ports' must be unique!")

        #ports are positive integers
        if not all(isinstance(p, int) and p >= 0 for p in ports):
            raise ValueError("'ports' must be positive integers!")

        self.block = block
        self.ports = [0] if ports is None else ports 


    def set(self, values):
        """Sets the input ports of the reference block with values.

        Note
        ----
        If more values then ports, `zip` automatically stops iteration 
        after all ports. If more ports then values, `itertools.cycle` is 
        used to fill all the ports repeatedly.

        Parameters
        ----------
        values : list[obj], list[float]
            values to set at block input ports
        """
        for p, v in zip(self.ports, cycle(values)):
            self.block.set(p, v)


    def get(self):
        """Returns the values of the output ports of the reference block.
        
        Returns
        -------
        out : list[obj], list[float]
            values from block output ports
        """
        return [self.block.get(p) for p in self.ports]


    def to_dict(self):
        """Serialization into dict"""
        return {
            "block": id(self.block),
            "ports": self.ports
        }