#########################################################################################
##
##                                PORT REFERENCE CLASS 
##                              (utils/portreference.py)
##                              
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np


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
    ports : list[int, str]
        list of port indices or names
    """

    __slots__ = ["block", "ports"]


    def __init__(self, block=None, ports=None):

        #default port is '0'
        _ports = [0] if ports is None else ports 

        #type validation for ports
        if not isinstance(_ports, list):            
            raise ValueError(f"'ports' must be list[int, str] but is '{type(_ports)}'!")
        
        for p in _ports:

            #type validation for individual ports
            if not isinstance(p, (int, str)):
                raise ValueError(f"Port '{p}' must be (int, str) but is '{type(p)}'!")

            #validation for positive interger
            if isinstance(p, int) and p < 0:
                raise ValueError(f"Port '{p}' is int but must be positive!")
            
            #key existance validation for string ports
            if not (p in block.inputs or p in block.outputs):        
                raise ValueError(f"Port alias '{p}' not defined for Block {block}!")

        #port uniqueness validation
        if len(_ports) != len(set(_ports)):
            raise ValueError("'ports' must be unique!")

        self.block = block
        self.ports = _ports


    def __len__(self):
        """The number of ports managed by 'PortReference'"""
        return len(self.ports)


    def _validate_input_ports(self):
        """Check the existance of the input ports, specifically string port 
        aliases for the block inputs. Raises a ValueError if not existent.
        """
        for p in self.ports:
            if not p in self.block.inputs:
                raise ValueError(f"Input port '{p}' not defined for Block {self.block}!")


    def _validate_output_ports(self):
        """Check the existance of the output ports, specifically string port 
        aliases for the block inputs. Raises a ValueError if not existent.
        """
        for p in self.ports:
            if not p in self.block.outputs:
                raise ValueError(f"Output port '{p}' not defined for Block {self.block}!")


    def to(self, other):
        """Transfer the data between two `PortReference` instances, 
        in this direction `self` -> `other`. From outputs to inputs.

        Parameters
        ----------
        other : PortReference
            the `PortReference` instance to transfer data to from `self`
        """
        for a, b in zip(other.ports, self.ports):
            other.block.inputs[a] = self.block.outputs[b]


    def get_inputs(self):
        """Return the input values of the block at specified ports

        Returns
        -------
        out : list[float, obj]
            input values of block
        """
        return np.array([self.block.inputs[p] for p in self.ports])


    def set_inputs(self, vals):
        """Set the block inputs with values at specified ports

        Parameters
        ----------
        vals : list[float, obj]
            values to set at block input ports
        """
        for v, p in zip(vals, self.ports):
            self.block.inputs[p] = v


    def get_outputs(self):
        """Return the output values of the block at specified ports

        Returns
        -------
        out : list[float, obj]
            output values of block
        """
        return np.array([self.block.outputs[p] for p in self.ports])


    def set_outputs(self, vals):
        """Set the block outputs with values at specified ports

        Parameters
        ----------
        vals : list[float, obj]
            values to set at block output ports
        """
        for v, p in zip(vals, self.ports):
            self.block.outputs[p] = v
        

    def to_dict(self):
        """Serialization into dict"""
        return {
            "block": id(self.block),
            "ports": self.ports
        }