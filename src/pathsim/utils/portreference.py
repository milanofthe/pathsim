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

    def __init__(self, block=None, ports=[0]):

        #type validation for ports
        if not isinstance(ports, list):            
            raise ValueError(f"'ports' must be 'list[int]' but is '{type(ports)}'!")

        #ports are positive integers
        if not all(isinstance(p, int) and p >= 0 for p in ports):
            raise ValueError("'ports' must be positive integers!")   

        #unique ports
        if len(ports) != len(set(ports)):
            raise ValueError("'ports' must be unique!")

        self.block = block
        self.ports = ports 


    def to(self, other):
        """Transfer the data between two `PortReference` instances, 
        in this direction `self` -> `other`.

        Parameters
        ----------
        other : PortReference
            the `PortReference` instance to transfer data to from `self`
        """
        for p_s, p_o in zip(self.ports, other.ports):
            other.block.set(p_o, self.block.get(p_s))


    def to_dict(self):
        """Serialization into dict"""
        return {
            "block": id(self.block),
            "ports": self.ports
        }