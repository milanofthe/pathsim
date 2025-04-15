from importlib import metadata

try:
    __version__ = metadata.version("pathsim")
except Exception:
    __version__ = "unknown"

from .simulation import Simulation
from .connection import Connection, Duplex
from .subsystem import Subsystem, Interface
