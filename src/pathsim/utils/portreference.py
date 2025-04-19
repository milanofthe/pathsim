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
    """This is a container class that holds a reference 
    to a block and a list of ports.

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
        self.block = block
        self.ports = [0] if ports is None else ports 


    def set(self, values):
        for p, v in zip(self.ports, cycle(values)):
            self.block.set(p, v)


    def get(self):
        return [self.block.get(p) for p in self.ports]


    def to_dict(self):
        return {
            "block": id(self.block),
            "ports": self.ports
        }