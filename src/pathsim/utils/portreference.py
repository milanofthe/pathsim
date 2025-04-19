#########################################################################################
##
##                                PORT REFERENCE CLASS 
##                              (utils/portreference.py)
##                              
##                                 Milan Rother 2025
##
#########################################################################################

# IMPORTS ===============================================================================

# no dependencies


# CLASS =================================================================================

class PortReference:

    def __init__(self, block=None, ports=None):

        self.block = block
        self.ports = [0] if ports is None else ports # <- default port is set here


    def set(self, *values):
        for p, v in zip(self.ports, values):
            self.block.set(p, v)


    def get(self):
        return [self.block.get(p) for p in self.ports]